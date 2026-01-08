import torch
import numpy as np
from tqdm import tqdm
import logging
import inspect

import federatedscope.register as register
from federatedscope.llm.reward.reward_model_implementations import (
    GPT2HarmlessRewardModel, GPT2HelpfulRewardModel)

logger = logging.getLogger(__name__)

# Global cache for reward models to avoid reloading them on every call
REWARD_MODELS = {}


def get_reward_models(device):
    """
    Loads and caches the reward models.
    """
    global REWARD_MODELS
    if 'harmless' not in REWARD_MODELS:
        logger.info("Loading Harmlessness Reward Model...")
        REWARD_MODELS['harmless'] = GPT2HarmlessRewardModel(device=device)
    if 'helpful' not in REWARD_MODELS:
        logger.info("Loading Helpfulness Reward Model...")
        REWARD_MODELS['helpful'] = GPT2HelpfulRewardModel(device=device)
    return REWARD_MODELS['harmless'], REWARD_MODELS['helpful']


def _get_or_compute_hhrl_scores(ctx):
    """
    A helper function that computes reward scores and caches them in the `ctx`
    to avoid redundant computation within the same evaluation round.
    """
    cache_key = f'{ctx.cur_split}_hhrl_scores'
    if hasattr(ctx, cache_key):
        return getattr(ctx, cache_key)

    eval_loader = getattr(ctx, f'{ctx.cur_split}_loader', None)
    if eval_loader is None:
        logger.warning(f"ctx.{ctx.cur_split}_loader is not available, "
                       f"skipping reward eval.")
        return {}

    harmless_reward_model, helpful_reward_model = get_reward_models(
        ctx.device)

    all_harmless_scores = []
    all_helpful_scores = []

    original_padding_side = ctx.tokenizer.padding_side
    original_pad_token = ctx.tokenizer.pad_token

    ctx.tokenizer.padding_side = 'left'
    if ctx.tokenizer.pad_token is None:
        ctx.tokenizer.pad_token = ctx.tokenizer.eos_token

    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1
    }

    # Limit the number of samples for evaluation to speed up
    # Default: evaluate on max 100 samples, or all if less than 100
    max_eval_samples = getattr(ctx.cfg.eval, 'max_samples_for_reward', 100)
    if max_eval_samples <= 0:
        max_eval_samples = float('inf')  # Evaluate on all samples
    
    total_samples_evaluated = 0
    should_limit = max_eval_samples != float('inf')

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating with Reward Models")):
        # Stop early if we've reached the max number of samples
        if should_limit and total_samples_evaluated >= max_eval_samples:
            break
        
        # Handle different data formats: RLHF uses win_input_ids/lose_input_ids, 
        # regular training uses input_ids
        if 'win_input_ids' in batch:
            # RLHF format: use win_input_ids for evaluation
            input_ids = batch['win_input_ids'].to(ctx.device)
            attention_mask = batch.get('win_attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(ctx.device)
        elif 'input_ids' in batch:
            # Regular format
            input_ids = batch['input_ids'].to(ctx.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(ctx.device)
        else:
            logger.warning(f"Batch {batch_idx} does not contain 'input_ids' or 'win_input_ids', skipping...")
            continue
        
        # Decode the entire input_ids to get the formatted prompt string
        # This is what the model sees as input.
        prompts = ctx.tokenizer.batch_decode(input_ids,
                                             skip_special_tokens=True)

        # Generate with or without attention_mask
        if attention_mask is not None:
            generated_ids = ctx.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=ctx.cfg.llm.max_new_token,
                **generation_kwargs)
        else:
            generated_ids = ctx.model.generate(
                input_ids=input_ids,
                max_new_tokens=ctx.cfg.llm.max_new_token,
                **generation_kwargs)
        
        completions = ctx.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        # The full text for the reward model is the generated text
        # The prompt for the reward model is the original input text
        # Limit the number of samples per batch if needed
        batch_size = len(completions)
        if should_limit and total_samples_evaluated + batch_size > max_eval_samples:
            # Only evaluate the remaining samples needed
            remaining = max_eval_samples - total_samples_evaluated
            completions = completions[:remaining]
            prompts = prompts[:remaining]
            batch_size = remaining

        harmless_scores = harmless_reward_model.get_rewards(completions,
                                                            prompts)
        helpful_scores = helpful_reward_model.get_rewards(completions,
                                                          prompts)

        all_harmless_scores.extend(harmless_scores)
        all_helpful_scores.extend(helpful_scores)
        
        total_samples_evaluated += batch_size
        
        # Stop if we've reached the limit
        if should_limit and total_samples_evaluated >= max_eval_samples:
            break

    ctx.tokenizer.padding_side = original_padding_side
    ctx.tokenizer.pad_token = original_pad_token

    results = {}
    if all_harmless_scores:
        results['avg_harmlessness'] = np.mean(all_harmless_scores)
        if should_limit:
            logger.info(f"Evaluated {len(all_harmless_scores)} samples for harmlessness (limited from full dataset)")
    if all_helpful_scores:
        results['avg_helpfulness'] = np.mean(all_helpful_scores)
        if should_limit:
            logger.info(f"Evaluated {len(all_helpful_scores)} samples for helpfulness (limited from full dataset)")

    setattr(ctx, cache_key, results)
    return results


# --- Metric 1: Harmlessness ---
def eval_harmlessness(ctx, **kwargs):
    scores = _get_or_compute_hhrl_scores(ctx)
    return scores.get('avg_harmlessness', 0.0)


def register_harmlessness_metric(types):
    if 'avg_harmlessness' in types:
        return 'avg_harmlessness', eval_harmlessness, True
    return None


# --- Metric 2: Helpfulness ---
def eval_helpfulness(ctx, **kwargs):
    scores = _get_or_compute_hhrl_scores(ctx)
    return scores.get('avg_helpfulness', 0.0)


def register_helpfulness_metric(types):
    if 'avg_helpfulness' in types:
        return 'avg_helpfulness', eval_helpfulness, True
    return None


# Register both metrics with the framework
register.register_metric('avg_harmlessness', register_harmlessness_metric)
register.register_metric('avg_helpfulness', register_helpfulness_metric)
