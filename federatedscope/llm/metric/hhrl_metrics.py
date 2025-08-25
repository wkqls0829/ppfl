import torch
import os
import numpy as np
from tqdm import tqdm
import logging

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

    # Directly get the data for the current split ('val' or 'test')
    current_data = getattr(ctx, f'{ctx.cur_split}_data', None)
    if current_data is None:
        logger.warning(f"ctx.{ctx.cur_split}_data is not available, "
                       f"skipping reward eval.")
        return {}

    harmless_reward_model, helpful_reward_model = get_reward_models(
        ctx.device)

    all_harmless_scores = []
    all_helpful_scores = []

    # Iterate over the correct data object
    for i in tqdm(range(len(current_data)),
                  desc="Evaluating with Reward Models"):
        sample = current_data[i]
        
        if 'prompt' not in sample:
            logger.warning(f"Sample {i} is missing the 'prompt' key. Skipping.")
            continue
        prompt = sample['prompt']

        if not hasattr(ctx.model, 'generate'):
            raise AttributeError(
                "The model in ctx does not have a `generate` method.")

        input_ids = ctx.tokenizer(
            prompt, return_tensors="pt").input_ids.to(ctx.device)
        generated_ids = ctx.model.generate(
            input_ids,
            max_new_tokens=ctx.cfg.llm.max_new_token,
            **ctx.cfg.llm.generation.kwargs)
        
        completion = ctx.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        harmless_score = harmless_reward_model.get_rewards([completion],
                                                           [prompt])[0]
        helpful_score = helpful_reward_model.get_rewards([completion],
                                                         [prompt])[0]

        all_harmless_scores.append(harmless_score)
        all_helpful_scores.append(helpful_score)

    results = {}
    if all_harmless_scores:
        results['avg_harmlessness'] = np.mean(all_harmless_scores)
    if all_helpful_scores:
        results['avg_helpfulness'] = np.mean(all_helpful_scores)

    # Cache the results in the context before returning
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

