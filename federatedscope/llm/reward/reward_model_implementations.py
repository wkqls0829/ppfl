import logging
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from federatedscope.llm.reward.reward_calculators import (
    cal_gpt2_harmless_probabilities, cal_gpt2_helpful_probabilities)

logger = logging.getLogger(__name__)

# A dedicated, self-contained function to load reward models
def _load_reward_model_and_tokenizer(model_name, device=None):
    """
    Loads a reward model and its tokenizer from Hugging Face.
    This is a simplified loader that does not depend on the main FS config.
    """
    logger.info(f"Loading reward model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device:
        model.to(device)
    model.eval()
    return tokenizer, model


class BaseRewardModel(object):
    def __init__(self, device: Optional[str] = None):
        self.device = device

    def get_rewards(self, texts: List[str],
                    prompts: Optional[List[str]] = None) -> List[float]:
        raise NotImplementedError


class GPT2HarmlessRewardModel(BaseRewardModel):
    """Reward model for harmlessness evaluation using a GPT-2 based model."""
    def __init__(self,
                 device: Optional[str] = None,
                 use_probabilities: bool = False):
        super().__init__(device)
        self.tokenizer, self.model = _load_reward_model_and_tokenizer(
            "liujch1998/gpt2-harmless-reward-model", device=self.device)
        self.use_probabilities = use_probabilities
        logger.info("GPT2 harmlessness reward model initialized")

    def get_rewards(self,
                    texts: List[str],
                    prompts: Optional[List[str]] = None) -> List[float]:
        if prompts is None or len(prompts) != len(texts):
            raise ValueError("Prompts must be provided for the GPT2 harmless"
                             " model")
        continuations = [
            t[len(p):].strip() if t.startswith(p) else t
            for p, t in zip(prompts, texts)
        ]
        probabilities, raw_scores = cal_gpt2_harmless_probabilities(
            prompts, continuations, self.model, self.tokenizer)
        return probabilities if self.use_probabilities else raw_scores


class GPT2HelpfulRewardModel(BaseRewardModel):
    """Reward model for helpfulness evaluation using a GPT-2 based model."""
    def __init__(self,
                 device: Optional[str] = None,
                 use_probabilities: bool = False):
        super().__init__(device)
        self.tokenizer, self.model = _load_reward_model_and_tokenizer(
            "liujch1998/gpt2-helpful-reward-model", device=self.device)
        self.use_probabilities = use_probabilities
        logger.info("GPT2 helpfulness reward model initialized")

    def get_rewards(self,
                    texts: List[str],
                    prompts: Optional[List[str]] = None) -> List[float]:
        if prompts is None or len(prompts) != len(texts):
            raise ValueError("Prompts must be provided for the GPT2 helpful"
                             " model")
        continuations = [
            t[len(p):].strip() if t.startswith(p) else t
            for p, t in zip(prompts, texts)
        ]
        probabilities, raw_scores = cal_gpt2_helpful_probabilities(
            prompts, continuations, self.model, self.tokenizer)
        return probabilities if self.use_probabilities else raw_scores

