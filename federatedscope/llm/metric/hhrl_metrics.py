import torch
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

    eval_loader = getattr(ctx, f'{ctx.cur_split}_loader', None)
    if eval_loader is None:
        logger.warning(f"ctx.{ctx.cur_split}_loader is not available, "
                       f"skipping reward eval.")
        return {}

    harmless_reward_model, helpful_reward_model = get_reward_models(
        ctx.device)

    all_harmless_scores = []
    all_helpful_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating with Reward Models"):
        # The dataloader provides tokenized inputs. We need to decode them
        # back to strings to get the prompt.
        input_ids = batch['input_ids'].to(ctx.device)
        
        # Decode the entire input_ids to get the formatted prompt string
        # This is what the model sees as input.
        prompts = ctx.tokenizer.batch_decode(input_ids,
                                             skip_special_tokens=True)

        if not hasattr(ctx.model, 'generate'):
            raise AttributeError(
                "The model in ctx does not have a `generate` method.")

        attention_mask = batch['attention_mask'].to(ctx.device)
        
        generated_ids = ctx.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=ctx.cfg.llm.max_new_token,
            **ctx.cfg.llm.generation.kwargs)
        
        completions = ctx.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        # The full text for the reward model is the generated text
        # The prompt for the reward model is the original input text
        harmless_scores = harmless_reward_model.get_rewards(completions,
                                                            prompts)
        helpful_scores = helpful_reward_model.get_rewards(completions,
                                                          prompts)

        all_harmless_scores.extend(harmless_scores)
        all_helpful_scores.extend(helpful_scores)

    results = {}
    if all_harmless_scores:
        results['avg_harmlessness'] = np.mean(all_harmless_scores)
    if all_helpful_scores:
        results['avg_helpfulness'] = np.mean(all_helpful_scores)

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
