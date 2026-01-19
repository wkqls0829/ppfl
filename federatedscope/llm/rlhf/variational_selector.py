"""
Variational Selector for RLHF
Uses VPL-trained variational encoder to sample client-specific z and make conditional choices.
"""
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from federatedscope.llm.dataset.llm_dataset import LLMComparisonDataset, DefaultToken
from federatedscope.llm.model.variational_encoder import VariationalEncoder
from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP

logger = logging.getLogger(__name__)


def cal_acc(logits, labels, choices):
    """Calculate accuracy for binary choice task."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx
    
    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)
    
    return new_labels, new_logits, predicted, predicted.eq(new_labels).sum().item()


def extract_preference_features_for_variational(model, tokenizer, list_data_dict, prompt_template, 
                                                choices, device='cuda:0', use_feature_difference=True):
    """
    Extract preference features from data for variational encoder.
    
    Args:
        model: LLM model (selector model)
        tokenizer: Tokenizer
        list_data_dict: List of data samples with output_A and output_B
        prompt_template: Prompt template for comparison
        choices: Choice token indices [A_token, B_token]
        device: Device to run on
        use_feature_difference: Whether to use embedding difference
        
    Returns:
        features: Extracted preference features (batch, feature_dim)
        hidden_states_list: List of hidden states for each sample
    """
    # Create dataset
    dataset = LLMComparisonDataset(
        list_data_dict,
        tokenizer,
        prompt_input=prompt_template,
        prompt_no_input=prompt_template,
        output_A="output_A",
        output_B="output_B",
        choice="fake_choice",  # Temporary choice for dataset creation
    )
    
    dataloader = DataLoader(dataset, batch_size=4)
    
    features_list = []
    hidden_states_list = []
    
    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(dataloader, desc="Extracting preference features"):
            # Get both win and lose inputs
            win_input_ids = data_batch["win_input_ids"].to(device)
            win_attention_mask = data_batch["win_attention_mask"].to(device)
            lose_input_ids = data_batch["lose_input_ids"].to(device)
            lose_attention_mask = data_batch["lose_attention_mask"].to(device)
            
            # Get hidden states
            win_outputs = model(input_ids=win_input_ids, attention_mask=win_attention_mask, output_hidden_states=True)
            lose_outputs = model(input_ids=lose_input_ids, attention_mask=lose_attention_mask, output_hidden_states=True)
            
            win_hidden = win_outputs.hidden_states[-1]  # Last layer hidden states
            lose_hidden = lose_outputs.hidden_states[-1]
            
            batch_size = win_hidden.shape[0]
            hidden_dim = win_hidden.shape[-1]
            
            # Extract embeddings at choice positions
            A_token, B_token = choices[0], choices[1]
            
            for b in range(batch_size):
                # For win (output_A) - find A token
                win_labels = win_input_ids[b]
                win_h = win_hidden[b]  # (seq_len, hidden_dim)
                
                # For lose (output_B) - find B token  
                lose_labels = lose_input_ids[b]
                lose_h = lose_hidden[b]  # (seq_len, hidden_dim)
                
                # Find choice token positions
                win_A_pos = (win_labels[1:] == A_token)  # Shift by 1 for alignment
                lose_B_pos = (lose_labels[1:] == B_token)
                
                if win_A_pos.any():
                    chosen_emb = win_h[:-1][win_A_pos].mean(dim=0)  # Mean over A positions
                else:
                    chosen_emb = win_h.mean(dim=0)  # Fallback to mean
                
                if lose_B_pos.any():
                    rejected_emb = lose_h[:-1][lose_B_pos].mean(dim=0)  # Mean over B positions
                else:
                    rejected_emb = lose_h.mean(dim=0)  # Fallback to mean
                
                # Compute difference: chosen - rejected
                feature_diff = chosen_emb - rejected_emb
                
                if use_feature_difference:
                    # Use [chosen, rejected, difference] for richer representation
                    feature = torch.cat([chosen_emb, rejected_emb, feature_diff], dim=0)  # (hidden_dim * 3,)
                else:
                    # Use only difference
                    feature = feature_diff  # (hidden_dim,)
                
                features_list.append(feature)
                hidden_states_list.append({
                    'chosen_emb': chosen_emb,
                    'rejected_emb': rejected_emb,
                    'difference': feature_diff
                })
    
    features = torch.stack(features_list, dim=0)  # (batch, feature_dim)
    return features, hidden_states_list


def variational_better_response(list_data_dict, selector_model, selector_tokenizer, 
                                variational_encoder, feature_extractor, prompt_template,
                                choices, device='cuda:0', use_feature_difference=True,
                                num_samples=1):
    """
    Use variational encoder to sample z and make conditional choices.
    
    Args:
        list_data_dict: List of data samples with output_A and output_B
        selector_model: Trained selector model (binary choice model)
        selector_tokenizer: Tokenizer
        variational_encoder: Trained VariationalEncoder or VariationalEncoderGP
        feature_extractor: Feature extractor network (from VPLRewardChoiceTrainer)
        prompt_template: Prompt template for comparison
        choices: Choice token indices [A_token, B_token]
        device: Device to run on
        use_feature_difference: Whether to use embedding difference
        num_samples: Number of z samples to draw per sample (for averaging)
        latent_projection: Projection layer from z to choice logits (from VPLRewardChoiceTrainer)
        
    Returns:
        list_data_dict: Updated with "choice" field (0 for A, 1 for B)
    """
    logger.info("Extracting preference features for variational selection...")
    
    # Extract preference features
    features, hidden_states_list = extract_preference_features_for_variational(
        selector_model, selector_tokenizer, list_data_dict, prompt_template,
        choices, device, use_feature_difference
    )
    
    # Process features through feature extractor if provided
    if feature_extractor is not None:
        features = feature_extractor(features.to(device))
    
    logger.info(f"Extracted features shape: {features.shape}")
    
    # Sample z from posterior q(z|x) using variational encoder
    variational_encoder.eval()
    selector_model.eval()
    
    predicted_indices = []
    
    with torch.no_grad():
        # Encode to get posterior parameters
        mu, logvar = variational_encoder.encode(features.to(device))
        
        # Sample z multiple times and average predictions
        all_predictions = []
        
        for sample_idx in range(num_samples):
            # Sample z from posterior
            z = variational_encoder.reparameterize(mu, logvar)  # (batch, latent_dim)
            
            # Create dataset for binary choice
            dataset = LLMComparisonDataset(
                list_data_dict,
                selector_tokenizer,
                prompt_input=prompt_template,
                prompt_no_input=prompt_template,
                output_A="output_A",
                output_B="output_B",
                choice="fake_choice",
            )
            dataloader = DataLoader(dataset, batch_size=4)
            
            batch_predictions = []
            z_idx = 0  # Track z index across batches
            
            for batch_idx, data_batch in enumerate(tqdm(dataloader, desc=f"Variational selection (sample {sample_idx+1}/{num_samples})")):
                win_input_ids = data_batch["win_input_ids"].to(device)
                win_labels = data_batch["win_labels"].to(device)
                win_attention_mask = data_batch["win_attention_mask"].to(device)
                lose_input_ids = data_batch["lose_input_ids"].to(device)
                lose_labels = data_batch["lose_labels"].to(device)
                lose_attention_mask = data_batch["lose_attention_mask"].to(device)
                
                batch_size = win_input_ids.shape[0]
                z_batch = z[z_idx:z_idx + batch_size]  # Get z for this batch
                z_idx += batch_size
                
                # Get logits for both responses
                win_outputs = selector_model(input_ids=win_input_ids, attention_mask=win_attention_mask)
                lose_outputs = selector_model(input_ids=lose_input_ids, attention_mask=lose_attention_mask)
                
                win_logits = win_outputs.logits
                lose_logits = lose_outputs.logits
                
                # Condition on z: project z to choice logits and add to model logits
                # Option 1: Add z-conditioned bias to choice token logits
                # Option 2: Scale logits based on z
                # We'll use Option 1: project z to choice space and add as bias
                
                # Project z to choice logits (similar to VPLRewardChoiceTrainer)
                # latent_projection should be provided as a separate parameter
                latent_dim = z_batch.shape[-1]
                num_choices = len(choices)
                
                # Use provided latent_projection if available, otherwise create temporary one
                if latent_projection is not None:
                    z_projection = latent_projection(z_batch)  # (batch, num_choices)
                else:
                    # Create temporary projection (should be loaded from trainer)
                    temp_projection = torch.nn.Linear(latent_dim, num_choices).to(device)
                    torch.nn.init.normal_(temp_projection.weight, mean=0.0, std=0.01)
                    torch.nn.init.zeros_(temp_projection.bias)
                    z_projection = temp_projection(z_batch)
                
                # Get logits at choice positions
                shift_win_logits = win_logits[..., :-1, :].contiguous()
                shift_lose_logits = lose_logits[..., :-1, :].contiguous()
                shift_win_labels = win_labels[..., 1:].contiguous()
                shift_lose_labels = lose_labels[..., 1:].contiguous()
                
                # Extract choice logits
                win_choice_logits = shift_win_logits[..., choices]  # (batch, seq_len-1, num_choices)
                lose_choice_logits = shift_lose_logits[..., choices]
                
                # Find choice token positions
                A_token, B_token = choices[0], choices[1]
                
                batch_choices = []
                for b in range(batch_size):
                    # Find A and B positions in win (output_A)
                    win_A_pos = (shift_win_labels[b] == A_token)
                    win_B_pos = (shift_win_labels[b] == B_token)
                    
                    # Find A and B positions in lose (output_B)
                    lose_A_pos = (shift_lose_labels[b] == A_token)
                    lose_B_pos = (shift_lose_labels[b] == B_token)
                    
                    # Get logits at choice positions
                    if win_A_pos.any():
                        win_A_logit = win_choice_logits[b, win_A_pos, 0].mean()  # Choice 0 = A
                    else:
                        win_A_logit = win_choice_logits[b, :, 0].mean()
                    
                    if lose_B_pos.any():
                        lose_B_logit = lose_choice_logits[b, lose_B_pos, 1].mean()  # Choice 1 = B
                    else:
                        lose_B_logit = lose_choice_logits[b, :, 1].mean()
                    
                    # Add z-conditioned bias
                    z_bias_A = z_projection[b, 0]  # Bias for choice A
                    z_bias_B = z_projection[b, 1]  # Bias for choice B
                    
                    # Compare: A (win) vs B (lose) with z conditioning
                    score_A = win_A_logit + z_bias_A
                    score_B = lose_B_logit + z_bias_B
                    
                    # Choose based on scores
                    if score_A > score_B:
                        choice = 0  # A is better
                    else:
                        choice = 1  # B is better
                    
                    batch_choices.append(choice)
                
                batch_predictions.extend(batch_choices)
            
            all_predictions.append(batch_predictions)
        
        # Average predictions across multiple z samples
        if num_samples > 1:
            all_predictions = torch.tensor(all_predictions)  # (num_samples, batch)
            predicted_indices = all_predictions.mode(dim=0)[0].tolist()  # Majority vote
        else:
            predicted_indices = all_predictions[0]
    
    # Update data with choices
    for choice, sample in zip(predicted_indices, list_data_dict):
        sample["choice"] = choice
        sample.pop("fake_choice", None)
    
    logger.info(f"Variational selection completed. Choices: {sum(predicted_indices)}/{len(predicted_indices)} chose B")
    
    return list_data_dict
