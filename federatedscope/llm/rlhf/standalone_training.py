import logging
import os
import random
import json
from tqdm import tqdm
import numpy as np
import gc
import copy
from itertools import combinations

import torch
from torch.utils.data import DataLoader

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.data import ClientData
from federatedscope.llm.dataloader import LLMDataCollator
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.dataset.llm_dataset import (
    DefaultToken,
    LLMDataset,
    LLMComparisonDataset,
)
from federatedscope.llm.trainer.reward_trainer import (
    DPORewardTrainer,
    _get_batch_logps,
    dpo_loss,
)
from federatedscope.core.auxiliaries.utils import add_prefix_to_path

logger = logging.getLogger(__name__)


@torch.no_grad()
def cal_acc(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    # print(new_logits)
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)

    return new_labels, new_logits, predicted, predicted.eq(
        new_labels).sum().item()


def get_rlhf_prompts_dataset(config):
    dataset_name, _ = config.data.type.split("@")

    if dataset_name.lower() == "reddit-tldr-rlhf":
        from federatedscope.llm.dataloader.reddit_tldr import (
            load_human_finetuning_dataset,
            TLDR_PROMPT_DICT,
        )

        data_root = os.path.join(config.data.root, "reddit-tldr-comparison")
        list_train_prompts, _, _ = load_human_finetuning_dataset(
            data_root,
            tokenizer=None,
            rlhf=True,
            max_num_test=1000,
            raw_no_prompt=True)
        generation_prompt = TLDR_PROMPT_DICT["summary"]
        selector_prompt = TLDR_PROMPT_DICT["summary_cmp"]

    elif dataset_name.lower() == "hh-rlhf":
        from federatedscope.llm.dataloader.hh_rlhf import (
            load_hh_rlhf_for_rlhf,
            HH_RLHF_PROMPT_DICT,
        )
        data_root = os.path.join(config.data.root, "hh-rlhf")

        list_train_prompts, _, _ = load_hh_rlhf_for_rlhf(
            data_root,
            config,
            max_num_test=1000,
            raw_no_prompt=True,
        )

        generation_prompt = HH_RLHF_PROMPT_DICT["generation"]
        selector_prompt = HH_RLHF_PROMPT_DICT["comparison"]

    elif dataset_name.lower() == "shp-rlhf":
        from federatedscope.llm.dataloader.shp import \
            load_rlhf_dataset, SHP_PROMPT_DICT

        data_root = os.path.join(config.data.root, 'shp')
        list_train_prompts, _, _ = load_rlhf_dataset(data_root,
                                                     tokenizer=None,
                                                     max_num_test=1000)
        generation_prompt = SHP_PROMPT_DICT["shp"]
        selector_prompt = SHP_PROMPT_DICT["shp_cmp"]

    elif dataset_name.lower() == "shp-safe":
        from federatedscope.llm.dataloader.shp import \
            load_safe_dataset, SHP_PROMPT_DICT

        data_root = os.path.join(config.data.root, 'shp')
        list_train_prompts, _, _ = load_safe_dataset()
        generation_prompt = SHP_PROMPT_DICT["shp"]
        selector_prompt = SHP_PROMPT_DICT["shp_cmp"]

    return (data_root, list_train_prompts, generation_prompt, selector_prompt)


def get_input_data(list_data_dict, w=10):
    for left in tqdm(range(0, len(list_data_dict), w)):
        yield list_data_dict[left:left + w]


class RLHF_finetuning:
    """
    Implementation of RLHF server
    """
    def __init__(
        self,
        model,
        tokenizer,
        config=None,
        selector_model=None,
        selector_tokenizer=None,
        generator_tokenizer=None,
        device="cpu",
        **kwargs,
    ):
        # obtain RLHF input data
        (
            self.data_root,
            self.list_train_prompts,
            self.generation_prompt,
            self.selector_prompt,
        ) = get_rlhf_prompts_dataset(config)

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.selector_model = selector_model
        self.selector_tokenizer = selector_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.device = device
        self._monitor = Monitor(config, monitored_object=self)
        
        # Client-specific average z values (computed once and reused)
        # {client_id: z_mu_tensor}
        self.client_average_z_dict = None
        self.num_clients = getattr(config.federate, 'client_num', 10)  # Default to 10 for hh-rlhf

    def load_pairwise_data(self):
        # Name of a file saving the generated texts of original model
        _, model_name = self.config.model.type.split("@")[0].split('/', 1)
        dataset_name, _ = self.config.data.type.split("@")
        num_comp = max(2, self.config.llm.num_completions)
        gen_fp = os.path.join(
            self.data_root,
            f"rlhf_pair_data_{model_name}_{dataset_name}_{num_comp}.json")

        # Check if VPL model is being used (for conditional generation)
        use_variational_generation = getattr(self.config.llm, 'rlhf_use_variational_generation', False)
        is_vpl_model = False
        
        if use_variational_generation:
            # Check if selector checkpoint has VPL components
            selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
            if selector_ckpt_path is None:
                selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
            
            if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                try:
                    from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                    variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                        selector_ckpt_path, self.config, device=self.device
                    )
                    if variational_encoder is not None and feature_extractor is not None:
                        is_vpl_model = True
                        logger.info("VPL model detected. Will use conditional generation with client-specific z.")
                except Exception as e:
                    logger.debug(f"Could not load VPL components: {e}. Using standard generation.")
        
        # Assign client_id to prompts only for VPL models
        if is_vpl_model:
            # Divide prompts evenly among clients
            num_clients = self.num_clients
            prompts_with_client_id = []
            for idx, prompt_data in enumerate(self.list_train_prompts):
                # Assign client_id (1-indexed, matching federated setup)
                client_id = (idx % num_clients) + 1
                prompt_data_with_id = copy.deepcopy(prompt_data)
                prompt_data_with_id['client_id'] = client_id
                prompts_with_client_id.append(prompt_data_with_id)
            
            logger.info(f"Assigned {len(prompts_with_client_id)} prompts to {num_clients} clients "
                       f"for conditional generation")
        else:
            # For non-VPL models, use prompts as-is (no client_id)
            prompts_with_client_id = self.list_train_prompts
            logger.info("Non-VPL model detected. Using standard generation (no client_id assignment).")

        if os.path.exists(gen_fp):
            # load the file with generated responses
            list_pairwise_data = json.load(open(gen_fp, "r"))
            logger.info("Successfully loaded the generated text "
                        f"from {gen_fp}")
            # For VPL models, ensure client_id is present in loaded data
            if is_vpl_model and len(list_pairwise_data) > 0 and 'client_id' not in list_pairwise_data[0]:
                logger.warning("Loaded pairwise data does not have client_id. "
                              "Regenerating with client_id assignment.")
                list_pairwise_data = None
        
        if not os.path.exists(gen_fp) or list_pairwise_data is None:
            # generate the output
            if is_vpl_model:
                logger.info("The generated text file does not exist or needs regeneration. "
                           "Create a new one with client_id assignment (VPL model).")
            else:
                logger.info("The generated text file does not exist. "
                           "Create a new one (standard generation).")
            list_pairwise_data = self._generate_pairwise_data(
                prompts_with_client_id,  # Use prompts with/without client_id based on is_vpl_model
                self.model,
                self.generator_tokenizer,
                self.generation_prompt,
                max_new_tokens=self.config.llm.max_new_token,
                num_completions=self.config.llm.num_completions)

            # save the data to a file
            json.dump(list_pairwise_data, open(gen_fp, "w"))
            logger.info("The generation process is done, and save "
                        f"to {gen_fp}.")
            
            # Visualize client-specific average z values after generation (for VPL models)
            if is_vpl_model and self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                try:
                    from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z
                    import numpy as np
                    import torch
                    
                    logger.info("Visualizing client-specific average z values used for generation...")
                    
                    # Prepare z values and client labels
                    z_values_list = []
                    client_labels_list = []
                    orthogonal_labels_list = []
                    
                    for client_id, z_mu in self.client_average_z_dict.items():
                        if isinstance(z_mu, torch.Tensor):
                            z_np = z_mu.cpu().numpy()
                        else:
                            z_np = np.array(z_mu)
                        z_values_list.append(z_np)
                        client_labels_list.append(client_id)
                        
                        # Assign orthogonal label based on client_id (first half = harmlessness, second half = helpfulness)
                        num_clients = self.num_clients
                        split_point = num_clients // 2
                        if client_id <= split_point:
                            orthogonal_labels_list.append(0)  # Harmlessness
                        else:
                            orthogonal_labels_list.append(1)  # Helpfulness
                    
                    if len(z_values_list) > 0:
                        z_array = np.array(z_values_list)
                        
                        # Visualize
                        output_dir = self.config.outdir
                        wandb_project = getattr(self.config.wandb, 'name_project', None)
                        
                        visualize_cross_client_z(
                            z_values=z_array,
                            client_labels=client_labels_list,
                            orthogonal_labels=orthogonal_labels_list,
                            orthogonal_prototypes=None,
                            round_num=-1,  # Use -1 to indicate "before training" / "generation"
                            output_dir=output_dir,
                            wandb_project=wandb_project
                        )
                        logger.info(f"Visualized {len(z_values_list)} client-specific average z values used for generation")
                        
                        # Log to WandB
                        if self.config.wandb.use and self.config.wandb.online_track:
                            try:
                                import wandb
                                generation_tsne_path = os.path.join(output_dir, 'cross_client_z_tsne_generation.png')
                                if output_dir and os.path.exists(generation_tsne_path):
                                    wandb.log({
                                        'visualization/client_average_z_tsne_generation': wandb.Image(generation_tsne_path)
                                    }, step=0)
                                logger.info("Logged client average z t-SNE visualization to WandB (generation phase)")
                            except Exception as e:
                                logger.warning(f"Failed to log client average z visualization to WandB: {e}")
                except Exception as e:
                    logger.warning(f"Failed to visualize client average z values: {e}")

        return list_pairwise_data

    def load_selector_preference_data(self, saveto, early_exiting=False):
        # This file save selector's choices
        fp = os.path.join(self.data_root, f"generated_choose_{saveto}.json")

        if os.path.exists(fp):
            list_preference_data = json.load(open(fp, "r"))

        else:
            list_pairwise_data = self.load_pairwise_data()

            # choose the better one based on the given output
            logger.info("Select the better response.")
            
            # Check if using variational selection
            use_variational_selection = getattr(self.config.llm, 'rlhf_use_variational_selection', False)
            
            if use_variational_selection:
                logger.info("Using variational selection with VPL posterior.")
                from federatedscope.llm.rlhf.variational_selector import variational_better_response
                from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                
                # Load VPL components from selector checkpoint
                selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
                if selector_ckpt_path is None:
                    # Try to infer from selector config
                    selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
                
                variational_encoder = None
                feature_extractor = None
                latent_projection = None
                z_to_embedding = None
                
                if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                    logger.info(f"Loading VPL components from {selector_ckpt_path}")
                    variational_encoder, feature_extractor, latent_projection, z_to_embedding = load_vpl_components_from_checkpoint(
                        selector_ckpt_path, self.config, device=self.device
                    )
                
                if variational_encoder is None:
                    logger.warning("Could not load VPL components. Falling back to standard selection.")
                    use_variational_selection = False
            
            if use_variational_selection:
                # Use variational selection
                choices = [self.selector_tokenizer(f": {c}")["input_ids"][-1] for c in ["A", "B"]]
                # Check if data already has z values (from previous rounds)
                use_provided_z = any('z' in sample for sample in list_pairwise_data)
                list_preference_data = variational_better_response(
                    list_pairwise_data,
                    self.selector_model,
                    self.selector_tokenizer,
                    variational_encoder,
                    feature_extractor,
                    self.selector_prompt,
                    choices,
                    device=self.device,
                    use_feature_difference=getattr(self.config.llm, 'vpl_use_feature_difference', True),
                    num_samples=getattr(self.config.llm, 'rlhf_variational_num_samples', 1),
                    latent_projection=latent_projection,
                    z_to_embedding=z_to_embedding,
                    use_provided_z=use_provided_z  # Use z from data if available
                )
            else:
                # Use standard selection
                list_preference_data = self._choose_better_response(
                    list_pairwise_data,
                    self.selector_model,
                    self.selector_tokenizer,
                self.selector_prompt,
            )
            logger.info(list_preference_data[0])
            # save the choice to a file
            json.dump(list_preference_data, open(fp, "w"))
            logger.info(f"Save the selection results to file {fp}")

            if early_exiting:
                # For choosing the answer
                exit(0)

        return list_preference_data

    def _compute_client_average_z_from_training_data(self, selector_ckpt_path):
        """
        Compute client-specific average z from training data using selector model.
        This is used when client average z is not available in checkpoint.
        
        Args:
            selector_ckpt_path: Path to selector checkpoint
        """
        try:
            from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
            import torch
            
            logger.info("Computing client-specific average z from training data...")
            
            # Load VPL components
            variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                selector_ckpt_path, self.config, device=self.device
            )
            
            if variational_encoder is None or feature_extractor is None:
                logger.warning("Cannot compute client average z: VPL components not available")
                return
            
            # Group prompts by client_id
            client_prompts = {i: [] for i in range(1, self.num_clients + 1)}
            for prompt_data in self.list_train_prompts:
                client_id = prompt_data.get('client_id', None)
                if client_id is None:
                    # Assign client_id if not present
                    idx = self.list_train_prompts.index(prompt_data)
                    client_id = (idx % self.num_clients) + 1
                if 1 <= client_id <= self.num_clients:
                    client_prompts[client_id].append(prompt_data)
            
            # Compute average z for each client
            self.client_average_z_dict = {}
            variational_encoder.eval()
            feature_extractor.eval()
            
            with torch.no_grad():
                for client_id, prompts in client_prompts.items():
                    if len(prompts) == 0:
                        continue
                    
                    # Sample a subset of prompts for efficiency (max 100 per client)
                    max_samples = min(100, len(prompts))
                    sampled_prompts = prompts[:max_samples]
                    
                    # Extract features and compute z for each prompt
                    z_list = []
                    for prompt_data in sampled_prompts:
                        prompt_text = self.generation_prompt.format_map(prompt_data)
                        input_tokens = self.generator_tokenizer(
                            prompt_text,
                            padding=True,
                            add_special_tokens=True,
                            return_tensors="pt",
                        )
                        input_ids = input_tokens['input_ids'].to(self.device)
                        attention_mask = input_tokens['attention_mask'].to(self.device)
                        
                        # Get embeddings
                        if hasattr(self.model, 'get_input_embeddings'):
                            input_embeddings = self.model.get_input_embeddings()(input_ids)
                        else:
                            # Fallback: use selector model
                            input_embeddings = self.selector_model.get_input_embeddings()(input_ids)
                        
                        # Pool embeddings
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled_embeddings = (input_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                        
                        # Extract features and encode to z
                        features = feature_extractor(pooled_embeddings)
                        z_mu, z_logvar = variational_encoder.encode(features)
                        z_list.append(z_mu)
                    
                    if len(z_list) > 0:
                        # Average z for this client
                        z_stack = torch.stack(z_list)
                        avg_z = z_stack.mean(dim=0)  # (latent_dim,)
                        self.client_average_z_dict[client_id] = avg_z
                        logger.debug(f"Computed average z for client {client_id}: shape {avg_z.shape}")
            
            logger.info(f"Computed average z for {len(self.client_average_z_dict)} clients from training data")
            
        except Exception as e:
            logger.error(f"Failed to compute client average z from training data: {e}")
            self.client_average_z_dict = None

    def train(self, saveto=None, early_exiting=False):
        if saveto is None:
            _, saveto = os.path.split(self.config.federate.save_to)
        # The training data should be selector's preference data
        list_train_dict = self.load_selector_preference_data(
            saveto, early_exiting)

        # move selector model to cpu
        self.selector_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # load comparison dataset
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            self.tokenizer,
            prompt_input=self.generation_prompt,
            prompt_no_input=self.generation_prompt,
            output_A="output_A",
            output_B="output_B",
            choice="choice",
        )
        
        # Create DataLoader directly instead of using ClientData
        # to avoid initialization issues with ClientData's __init__
        from torch.utils.data import DataLoader
        from federatedscope.llm.dataloader import LLMRewardCollator
        
        data_collator = LLMRewardCollator(tokenizer=self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=self.config.dataloader.shuffle,
            num_workers=self.config.dataloader.num_workers,
            collate_fn=data_collator,
            pin_memory=self.config.dataloader.pin_memory,
        )
        
        # Create data dict compatible with trainer expectations
        data = {
            'train': train_dataloader,
            'val': None,
            'test': None,
        }

        # create DPO trainer
        self.trainer = DPORewardTrainer(
            self.model,
            data,
            self.device,
            self.config,
            only_for_eval=False,
            monitor=self._monitor,
        )

        # Load test data for evaluation
        # Use original hh-rlhf test data for evaluation
        # Directly load test data from datasets (don't rely on load_hh_rlhf_for_rlhf which returns None for test when raw_no_prompt=True)
        test_dataset = None
        try:
            # Load original test data with chosen/rejected pairs
            import datasets
            from federatedscope.llm.dataloader.hh_rlhf import parse_dialogue
            
            logger.info("Loading test data from hh-rlhf dataset for evaluation...")
            harmless_test = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split='test')
            helpful_test = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split='test')
            combined_test = datasets.concatenate_datasets([harmless_test, helpful_test])
            
            # Limit to max_samples_for_reward
            max_test_samples = getattr(self.config.eval, 'max_samples_for_reward', 30)
            if max_test_samples > 0 and len(combined_test) > max_test_samples:
                combined_test = combined_test.select(range(max_test_samples))
            
            # Convert to comparison format
            # Assign client_id only for VPL models (for conditional generation)
            list_test_dict = []
            use_variational_generation = getattr(self.config.llm, 'rlhf_use_variational_generation', False)
            is_vpl_model = False
            
            if use_variational_generation:
                # Check if selector checkpoint has VPL components
                selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
                if selector_ckpt_path is None:
                    selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
                
                if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                    try:
                        from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                        variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                            selector_ckpt_path, self.config, device=self.device
                        )
                        if variational_encoder is not None and feature_extractor is not None:
                            is_vpl_model = True
                    except Exception:
                        pass
            
            num_clients = self.num_clients if is_vpl_model else 1
            for idx, example in enumerate(combined_test):
                prompt, chosen = parse_dialogue(example['chosen'])
                _, rejected = parse_dialogue(example['rejected'])
                if prompt and chosen and rejected:
                    test_sample = {
                        'prompt': prompt,
                        'output_A': chosen,
                        'output_B': rejected,
                        'choice': 0,  # chosen (A) is better
                    }
                    # Assign client_id only for VPL models
                    if is_vpl_model:
                        client_id = (idx % num_clients) + 1
                        test_sample['client_id'] = client_id
                    list_test_dict.append(test_sample)
            
            if is_vpl_model:
                logger.info(f"Assigned {len(list_test_dict)} test samples to {num_clients} clients "
                           f"for conditional generation (VPL model)")
            else:
                logger.info(f"Loaded {len(list_test_dict)} test samples (standard generation, no client_id)")
            
            if len(list_test_dict) > 0:
                test_dataset = LLMComparisonDataset(
                    list_test_dict,
                    self.tokenizer,
                    prompt_input=self.generation_prompt,
                    prompt_no_input=self.generation_prompt,
                    output_A="output_A",
                    output_B="output_B",
                    choice="choice",
                )
                
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=self.config.dataloader.batch_size,
                    shuffle=False,  # Don't shuffle test data
                    num_workers=self.config.dataloader.num_workers,
                    collate_fn=data_collator,
                    pin_memory=self.config.dataloader.pin_memory,
                )
                
                # Update data dict with test loader
                data['test'] = test_dataloader
                # Update trainer's data
                self.trainer.data = data
                self.trainer.ctx.test_loader = test_dataloader
                logger.info(f"Loaded {len(list_test_dict)} test samples for evaluation")
            else:
                logger.warning("Test dataset is empty or could not be created. Test evaluation will be skipped.")
        except Exception as e:
            logger.error(f"Failed to load test data: {e}. Test evaluation will be skipped.")
        
        # Initialize z values storage for visualization
        z_values_list = []
        z_mu_list = []
        z_logvar_list = []
        
        # start training
        for r in range(self.config.federate.total_round_num):
            logger.info("----------- Starting a new RLHF training round "
                        f"(Round #{r}) -------------")
            sample_size, model_para_all, results = self.trainer.train()
            train_log_res = self._monitor.format_eval_res(results,
                                                          rnd=r,
                                                          role="Server",
                                                          return_raw=True)
            logger.info(train_log_res)
            
            # Filter train results: only keep loss, remove winrate and reward model scores
            train_log_res_filtered = train_log_res.copy()
            if 'Results_raw' in train_log_res_filtered:
                train_results = train_log_res_filtered['Results_raw']
                # Keep only loss-related metrics
                keys_to_remove = [
                    'train_helpfulness_winrate', 'train_harmlessness_winrate', 
                    'train_avg_winlose_rate', 'train_avg_helpfulness', 
                    'train_avg_harmlessness'
                ]
                for key in keys_to_remove:
                    train_results.pop(key, None)
                train_log_res_filtered['Results_raw'] = train_results
            
            # Save filtered train results to WandB (only loss)
            if self.config.wandb.use and self.config.wandb.online_track:
                self._monitor.save_formatted_results(train_log_res_filtered, save_file_name="")
            
            # Collect z values for visualization (if variational generation is enabled)
            if (hasattr(self.trainer, 'use_variational_generation') and 
                self.trainer.use_variational_generation and
                hasattr(self.trainer, 'variational_encoder') and
                self.trainer.variational_encoder is not None):
                # Collect z values from trainer
                try:
                    if hasattr(self.trainer, 'get_collected_z_values'):
                        z_vals = self.trainer.get_collected_z_values()
                        if z_vals is not None and len(z_vals) > 0:
                            if isinstance(z_vals, torch.Tensor):
                                z_vals = z_vals.cpu().numpy()
                            z_values_list.extend(z_vals)
                            logger.info(f"Round {r}: Collected {len(z_vals)} z values for visualization (total: {len(z_values_list)})")
                            # Clear collected z values after collecting
                            if hasattr(self.trainer, 'clear_collected_z_values'):
                                self.trainer.clear_collected_z_values()
                except Exception as e:
                    logger.debug(f"Could not collect z values in round {r}: {e}")
            
            # Evaluate on test split if available
            # Check both data dict and trainer's data dict
            test_available = (data.get('test') is not None) or (hasattr(self.trainer, 'data') and self.trainer.data.get('test') is not None)
            if test_available and (r + 1) % self.config.eval.freq == 0:
                logger.info("----------- Evaluating on test split -------------")
                # Ensure test_loader is set in ctx
                if hasattr(self.trainer, 'data') and 'test' in self.trainer.data:
                    self.trainer.ctx.test_loader = self.trainer.data['test']
                elif 'test' in data:
                    self.trainer.ctx.test_loader = data['test']
                
                test_results = self.trainer.evaluate(target_data_split_name="test")
                if test_results is None:
                    # If evaluate returns None, try to get eval_metrics from ctx
                    test_results = getattr(self.trainer.ctx, 'eval_metrics', {})
                
                if test_results:
                    test_log_res = self._monitor.format_eval_res(test_results,
                                                                 rnd=r,
                                                                 role="Server",
                                                                 return_raw=True)
                    logger.info(test_log_res)
                else:
                    logger.warning(f"Round {r+1}: Test evaluation returned no results.")
                    test_log_res = None
                
                # Save test results to WandB
                if test_log_res and self.config.wandb.use and self.config.wandb.online_track:
                    self._monitor.save_formatted_results(test_log_res, save_file_name="")
            elif (r + 1) % self.config.eval.freq == 0:
                logger.warning(f"Round {r+1}: Test data not available for evaluation. Skipping test evaluation.")
            
            # Visualize z values periodically (every 5 rounds or at the end)
            if (len(z_values_list) > 0 and 
                ((r + 1) % 5 == 0 or r == self.config.federate.total_round_num - 1)):
                try:
                    from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z
                    import numpy as np
                    import matplotlib.pyplot as plt
                    
                    # For standalone RL, we only have one "client" (the server)
                    # Create client labels (all 1 for standalone)
                    client_labels = [1] * len(z_values_list)
                    
                    # Convert to numpy array
                    z_array = np.array(z_values_list)
                    
                    # Visualize t-SNE
                    output_dir = self.config.outdir
                    wandb_project = getattr(self.config.wandb, 'name_project', None)
                    
                    visualize_cross_client_z(
                        z_values=z_array,
                        client_labels=client_labels,
                        orthogonal_labels=None,  # No orthogonal labels in standalone RL
                        orthogonal_prototypes=None,
                        round_num=r,
                        output_dir=output_dir,
                        wandb_project=wandb_project
                    )
                    logger.info(f"Round {r}: Generated t-SNE visualization with {len(z_values_list)} z values")
                    
                    # Log z statistics to WandB
                    if self.config.wandb.use and self.config.wandb.online_track:
                        try:
                            import wandb
                            
                            # Calculate statistics
                            z_mean = np.mean(z_array, axis=0)
                            z_std = np.std(z_array, axis=0)
                            z_norm = np.linalg.norm(z_array, axis=1)
                            
                            # Log statistics
                            wandb_metrics = {
                                f'z_stats/mean_norm': np.mean(z_norm),
                                f'z_stats/std_norm': np.std(z_norm),
                                f'z_stats/mean_dim_0': float(z_mean[0]) if len(z_mean) > 0 else 0.0,
                                f'z_stats/std_dim_0': float(z_std[0]) if len(z_std) > 0 else 0.0,
                                f'z_stats/num_values': len(z_values_list),
                            }
                            
                            # Log histogram of z norms
                            fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                            ax_hist.hist(z_norm, bins=30, alpha=0.7, edgecolor='black')
                            ax_hist.set_xlabel('||z|| (L2 norm)', fontsize=12)
                            ax_hist.set_ylabel('Frequency', fontsize=12)
                            ax_hist.set_title(f'Distribution of z Norms (Round {r})', fontsize=14)
                            ax_hist.grid(True, alpha=0.3)
                            plt.tight_layout()
                            
                            wandb_metrics[f'z_stats/z_norm_histogram'] = wandb.Image(fig_hist)
                            
                            wandb.log(wandb_metrics, step=r)
                            plt.close(fig_hist)
                            
                            logger.info(f"Round {r}: Logged z statistics to WandB")
                        except Exception as e:
                            logger.warning(f"Failed to log z statistics to WandB: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to visualize z values in round {r}: {e}")
            # Save the checkpoint
            if (r + 1) % self.config.federate.save_freq == 0:
                if saveto in self.config.federate.save_to:
                    path = add_prefix_to_path(f"{r + 1}_",
                                              self.config.federate.save_to)
                else:
                    path = add_prefix_to_path(f"{r + 1}_{saveto}_",
                                              self.config.federate.save_to)
                self.model.save_model(path=path, state=r)

    def _generate_pairwise_data(self,
                                list_data_dict,
                                model,
                                tokenizer,
                                prompt,
                                max_new_tokens=60,
                                num_completions=2):
        generate_kwargs = dict(
            top_p=1.0,
            temperature=0.7,
            do_sample=True,  # Must be True for num_return_sequences > 1
            # early_stopping=True,
            max_new_tokens=max_new_tokens,
            num_return_sequences=max(2, num_completions),
        )
        
        # Ensure do_sample is True when num_return_sequences > 1
        if generate_kwargs['num_return_sequences'] > 1 and not generate_kwargs.get('do_sample', False):
            generate_kwargs['do_sample'] = True
            logger.warning(f"num_return_sequences={generate_kwargs['num_return_sequences']} > 1, forcing do_sample=True")

        # Check if using z-dependent generation (Step 3)
        use_variational_generation = getattr(self.config.llm, 'rlhf_use_variational_generation', False)
        z_to_embedding = None
        
        # Load client average z once and reuse (if not already loaded)
        if use_variational_generation and self.client_average_z_dict is None:
            from federatedscope.llm.rlhf.load_vpl_components import (
                load_vpl_components_from_checkpoint,
                load_client_average_z_from_checkpoint
            )
            selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
            if selector_ckpt_path is None:
                selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
            
            if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                logger.info(f"Loading z_to_embedding and client average z for generation from {selector_ckpt_path}")
                _, _, _, z_to_embedding = load_vpl_components_from_checkpoint(
                    selector_ckpt_path, self.config, device=self.device
                )
                if z_to_embedding is None:
                    logger.warning("Failed to load z_to_embedding. Generation will not use z.")
                    use_variational_generation = False
                else:
                    # Load client-specific average z values from training data (compute once, reuse)
                    self.client_average_z_dict = load_client_average_z_from_checkpoint(
                        selector_ckpt_path, device=self.device
                    )
                    if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                        logger.info(f"Loaded average z for {len(self.client_average_z_dict)} clients. "
                                   f"Will use client-specific z for conditional generation.")
                    else:
                        logger.warning("No client average z found in checkpoint. Will compute from training data.")
                        # If not in checkpoint, compute from training data
                        self._compute_client_average_z_from_training_data(selector_ckpt_path)
        
        # Use stored client_average_z_dict
        client_average_z_dict = self.client_average_z_dict

        new_list_data_dict = []
        for input_data in get_input_data(list_data_dict):
            input_texts = [prompt.format_map(data) for data in input_data]
            input_text_tokens = tokenizer(
                input_texts,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            
            # Ensure model is on the correct device before tokenizing
            # Get model device (handle device_map='auto' case)
            if hasattr(model, 'device'):
                model_device = model.device
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
                model_device = model.base_model.device
            else:
                try:
                    model_device = next(model.parameters()).device
                except:
                    model_device = self.device
            
            # Move tokens to model device
            input_text_tokens_device = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in input_text_tokens.items()}
            
            input_ids = input_text_tokens_device['input_ids']
            attention_mask = input_text_tokens_device['attention_mask']

            # Step 3: Inject z into embeddings for generation
            if use_variational_generation and z_to_embedding is not None:
                # Priority 1: Use client-specific average z from training data (if available)
                # Priority 2: Use z from data (if already generated in previous rounds)
                # Priority 3: Use overall average z (for standalone mode)
                # Priority 4: Infer z from input (fallback)
                
                z = None
                batch_size = len(input_data)
                
                # Priority 1: Try to use client-specific average z
                if client_average_z_dict is not None and len(client_average_z_dict) > 0:
                    z_list = []
                    for data in input_data:
                        # Check if data has client_id
                        client_id = data.get('client_id', None)
                        if client_id is not None and client_id in client_average_z_dict:
                            # Use client-specific average z
                            z_list.append(client_average_z_dict[client_id])
                        else:
                            # Use overall average z (average of all clients)
                            all_z_mus = torch.stack(list(client_average_z_dict.values()))
                            overall_avg_z = all_z_mus.mean(dim=0)
                            z_list.append(overall_avg_z)
                    
                    if len(z_list) == batch_size:
                        z = torch.stack(z_list).to(self.device)
                        logger.debug(f"Using client-specific average z for generation: shape {z.shape}")
                
                # Priority 2: Try to get z from data (if already generated)
                if z is None and 'z' in input_data[0]:
                    z_list = [data.get('z', None) for data in input_data]
                    if all(z_val is not None for z_val in z_list):
                        z = torch.stack([
                            torch.tensor(z_val) if not isinstance(z_val, torch.Tensor) else z_val
                            for z_val in z_list
                        ]).to(self.device)
                        logger.debug(f"Using z from data: shape {z.shape}")
                
                # Priority 3: Use overall average z (if client_average_z_dict available but no client_id)
                if z is None and client_average_z_dict is not None and len(client_average_z_dict) > 0:
                    all_z_mus = torch.stack(list(client_average_z_dict.values()))
                    overall_avg_z = all_z_mus.mean(dim=0)  # (latent_dim,)
                    z = overall_avg_z.unsqueeze(0).repeat(batch_size, 1).to(self.device)  # (batch_size, latent_dim)
                    logger.debug(f"Using overall average z for generation: shape {z.shape}")
                
                # Priority 4: Infer z from input (fallback - not recommended)
                if z is None:
                    logger.warning("No client average z available. Inferring z from input (this is less accurate).")
                    # Use feature_extractor and variational_encoder to infer z
                    from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                    variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                        selector_ckpt_path, self.config, device=self.device
                    )
                    
                    if variational_encoder is not None and feature_extractor is not None:
                        # Ensure input_ids and attention_mask are on model device
                        input_ids = input_ids.to(model_device)
                        attention_mask = attention_mask.to(model_device)
                        # Get embeddings
                        input_embeddings = model.get_input_embeddings()(input_ids)
                        # Pool embeddings (mean over sequence length, masked)
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled_embeddings = (input_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                        # Extract features
                        features = feature_extractor(pooled_embeddings)
                        # Encode to z
                        z_mu, z_logvar = variational_encoder.encode(features)
                        z = z_mu  # Use mean during generation
                        logger.debug(f"Inferred z from input for generation: shape {z.shape}")
                
                if z is not None:
                    # Get model device (handle device_map='auto' case)
                    if hasattr(model, 'device'):
                        model_device = model.device
                    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
                        model_device = model.base_model.device
                    else:
                        try:
                            model_device = next(model.parameters()).device
                        except:
                            model_device = self.device
                    
                    # Ensure z is on model device
                    z = z.to(model_device)
                    
                    # Project z to embedding space and inject into input embeddings
                    z_embedding = z_to_embedding(z)  # (batch_size, embedding_dim)
                    z_embedding = z_embedding.unsqueeze(1)  # (batch_size, 1, embedding_dim)
                    input_embeddings = model.get_input_embeddings()(input_ids)
                    inputs_embeds = input_embeddings + z_embedding
                    
                    # Ensure inputs_embeds and attention_mask are on model device
                    inputs_embeds = inputs_embeds.to(model_device)
                    attention_mask = attention_mask.to(model_device)
                    
                    # Use inputs_embeds instead of input_ids for generation
                    output_ids = model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        **generate_kwargs
                    )
                else:
                    # Fallback to standard generation
                    # input_text_tokens_device already on model_device from above
                    output_ids = model.generate(**input_text_tokens_device, **generate_kwargs)
            else:
                # Standard generation without z
                # input_text_tokens_device already on model_device from above
                output_ids = model.generate(**input_text_tokens_device, **generate_kwargs)
            responses = tokenizer.batch_decode(output_ids,
                                               skip_special_tokens=True,
                                               ignore_tokenization_space=True)

            response_map = [[] for _ in input_data]
            for res in responses:
                for idx, input_text in enumerate(input_texts):
                    if input_text in res:
                        gen_res = res.replace(input_text, "").strip()
                        response_map[idx].append(gen_res.replace("</s>", ""))
                        # response_map[idx].append(
                        #     " " + gen_res.replace("</s>", ""))
                        break

            for i, data in enumerate(input_data):
                logger.info(data)
                for j, res in enumerate(response_map[i]):
                    logger.info(f'Generated {j}-th response: {res}')

                for output_A, output_B in combinations(response_map[i], 2):
                    new_data = copy.deepcopy(data)
                    new_data["output_A"] = output_A
                    new_data["output_B"] = output_B
                    new_list_data_dict.append(new_data)

        return new_list_data_dict

    @torch.no_grad()
    def _choose_better_response(self, list_data_dict, model, tokenizer,
                                prompt):
        choices = [tokenizer(f": {c}")["input_ids"][-1] for c in ["A", "B"]]
        logger.info(f'Choice indices: {choices}')

        for sample in list_data_dict:
            sample["fake_choice"] = random.choice([" A", " B"])

        token_dataset = LLMDataset(
            list_data_dict,
            tokenizer,
            prompt_input=prompt,
            prompt_no_input=prompt,
            output_tag="fake_choice",
        )
        dataloader = DataLoader(
            dataset=token_dataset,
            batch_size=10,
            shuffle=False,
            collate_fn=LLMDataCollator(tokenizer=tokenizer),
        )

        predicted_indices = []
        # Get model device
        model_device = self.device
        if hasattr(model, 'device'):
            model_device = model.device
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
            model_device = model.base_model.device
        else:
            try:
                model_device = next(model.parameters()).device
            except:
                model_device = self.device
        
        if hasattr(model, "adapter_names") is False or len(
                model.adapter_names) == 1:
            # No adapter or only one LoRA adapter
            for idx, data_batch in enumerate(tqdm(dataloader)):
                input_ids = data_batch["input_ids"].to(model_device)
                labels = data_batch["labels"].to(model_device)
                attention_mask = data_batch["attention_mask"].to(model_device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                _, _, predicted, _ = cal_acc(outputs.logits, labels, choices)
                predicted_indices += predicted.tolist()
        else:
            # More than one adapters (exclude "default" one)
            for idx, data_batch in enumerate(tqdm(dataloader)):
                input_ids = data_batch["input_ids"].to(model_device)
                labels = data_batch["labels"].to(model_device)
                attention_mask = data_batch["attention_mask"].to(model_device)
                collective_choices = []
                for name in model.adapter_names:
                    if name == "default":
                        continue
                    model.set_active_adapter(name)
                    model.eval()
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                    _, _, predicted, _ = cal_acc(outputs.logits, labels,
                                                 choices)
                    collective_choices.append(predicted.tolist())
                array = np.array(collective_choices).T
                predicted_indices += [
                    np.bincount(array[i]).argmax().item()
                    for i in range(len(array))
                ]

        for choice, sample in zip(predicted_indices, list_data_dict):
            sample["choice"] = choice
            sample.pop("fake_choice", None)

        return list_data_dict

    def dpo_better_response(self):
        # generate the output
        list_train_dict = self._generate_pairwise_data(
            self.list_train_prompts,
            self.model,
            self.generator_tokenizer,
            self.generation_prompt,
            max_new_tokens=self.config.llm.max_new_token,
        )

        return self._dpo_better_response(list_train_dict, self.model,
                                         self.tokenizer,
                                         self.generation_prompt)

    @torch.no_grad()
    def _dpo_better_response(self, list_data_dict, model, tokenizer, prompt):
        for sample in list_data_dict:
            sample["fake_choice"] = random.choice([0, 1])

        dataset = LLMComparisonDataset(
            list_data_dict,
            tokenizer,
            prompt_input=prompt,
            prompt_no_input=prompt,
            output_A="output_A",
            output_B="output_B",
            choice="fake_choice",
        )

        dataloader = DataLoader(dataset)

        # Get model device
        model_device = self.device
        if hasattr(model, 'device'):
            model_device = model.device
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
            model_device = model.base_model.device
        else:
            try:
                model_device = next(model.parameters()).device
            except:
                model_device = self.device

        predicted_indices = []
        for idx, data_batch in enumerate(tqdm(dataloader)):
            win_input_ids = data_batch["win_input_ids"].to(model_device)
            win_labels = data_batch["win_labels"].to(model_device)
            win_attention_mask = data_batch["win_attention_mask"].to(model_device)
            ref_win_outputs = model(
                disable_adapter=True,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask,
            )
            ref_win_logps = _get_batch_logps(ref_win_outputs.logits,
                                             win_labels,
                                             average_log_prob=False)
            policy_win_outputs = model(
                disable_adapter=False,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask,
            )
            policy_win_logps = _get_batch_logps(policy_win_outputs.logits,
                                                win_labels,
                                                average_log_prob=False)

            lose_input_ids = data_batch["lose_input_ids"].to(model_device)
            lose_labels = data_batch["lose_labels"].to(model_device)
            lose_attention_mask = data_batch["lose_attention_mask"].to(model_device)
            ref_lose_outputs = model(
                disable_adapter=True,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask,
            )
            ref_lose_logps = _get_batch_logps(ref_lose_outputs.logits,
                                              lose_labels,
                                              average_log_prob=False)
            policy_lose_outputs = model(
                disable_adapter=False,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask,
            )
            policy_lose_logps = _get_batch_logps(policy_lose_outputs.logits,
                                                 lose_labels,
                                                 average_log_prob=False)

            # DPO for reward calculation
            _, win_rewards, lose_rewards = dpo_loss(
                policy_win_logps,
                policy_lose_logps,
                ref_win_logps,
                ref_lose_logps,
                beta=1.0,
            )

            predicted = torch.where(
                win_rewards.cpu() > lose_rewards.cpu(),
                torch.zeros(len(win_input_ids)),
                torch.ones(len(win_input_ids)),
            )
            predicted_indices += predicted.tolist()

        for choice, sample in zip(predicted_indices, list_data_dict):
            sample["choice"] = choice
            sample.pop("fake_choice", None)

        return list_data_dict
