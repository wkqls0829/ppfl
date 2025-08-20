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


@torch.no_grad()
def eval_hhrl_reward(ctx, **kwargs):
    """
    Calculates harmlessness and helpfulness scores for generated text using
    reward models. This function is called by the trainer during evaluation.

    Args:
        ctx: The context object from the trainer, containing model, data, etc.

    Returns:
        A dictionary containing the average harmlessness and helpfulness scores.
    """
    harmless_reward_model, helpful_reward_model = get_reward_models(
        ctx.device)

    all_harmless_scores = []
    all_helpful_scores = []

    if not hasattr(ctx, 'eval_data') or ctx.eval_data is None:
        logger.warning("ctx.eval_data is not available, skipping reward eval.")
        return {}

    for i in tqdm(range(len(ctx.eval_data)),
                  desc="Evaluating with Reward Models"):
        sample = ctx.eval_data[i]
        
        # HH-RLHF data should have a 'prompt' key with the conversation history
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

    return results


# The registration function
def register_hhrl_reward_metric(types):
    if 'hhrl_reward' in types:
        return "hhrl_reward", eval_hhrl_reward, True


register.register_metric('hhrl_reward', register_hhrl_reward_metric)

