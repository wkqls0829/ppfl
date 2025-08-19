"""Implementation of specific reward models for text evaluation."""

import logging
from typing import List, Optional

from federatedscope.llm.model.model_builder import get_model_and_tokenizer
from federatedscope.llm.reward.reward_calculators import (
    cal_gpt2_harmless_probabilities, cal_gpt2_helpful_probabilities)

logger = logging.getLogger(__name__)


class BaseRewardModel(object):
    def __init__(self, device: Optional[str] = None):
        self.device = device

    def get_rewards(self, texts: List[str],
                    prompts: Optional[List[str]] = None) -> List[float]:
        raise NotImplementedError


class GPT2HarmlessRewardModel(BaseRewardModel):
    """Reward model for harmlessness evaluation using GPT-2 based model.

    Higher scores for more harmless content.
    """
    def __init__(self,
                 device: Optional[str] = None,
                 use_probabilities: bool = False):
        """Initialize the GPT-2 harmlessness reward model.

        Args:
            device (Optional[str]): Device to load the model on.
            use_probabilities (bool): Whether to return probabilities instead
            of raw scores.
        """
        super().__init__(device)

        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer(
            "gpt2-harmless", device=self.device)
        self.use_probabilities = use_probabilities

        logger.info("GPT2 harmlessness reward model initialized")

    def get_rewards(self,
                    texts: List[str],
                    prompts: Optional[List[str]] = None) -> List[float]:
        """Calculate harmlessness rewards for the given texts.

        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Prompt texts (required for this
            model)

        Returns:
            List[float]: Reward scores for each text

        Raises:
            ValueError: If prompts are not provided
        """
        if prompts is None or len(prompts) != len(texts):
            raise ValueError("Prompts must be provided and match the number "
                             "of texts for the GPT2 harmless model")

        # Extract continuations from texts
        continuations = [
            t[len(p):].strip() if t.startswith(p) else t
            for p, t in zip(prompts, texts)
        ]

        # Calculate harmlessness probabilities and scores
        probabilities, raw_scores = cal_gpt2_harmless_probabilities(
            prompts, continuations, self.model, self.tokenizer)

        # Return either probabilities or raw scores
        if self.use_probabilities:
            return probabilities
        else:
            return raw_scores


class GPT2HelpfulRewardModel(BaseRewardModel):
    """Reward model for helpfulness evaluation using GPT-2 based model.

    Higher scores for more helpful content.
    """
    def __init__(self,
                 device: Optional[str] = None,
                 use_probabilities: bool = False):
        """Initialize the GPT-2 helpfulness reward model.

        Args:
            device (Optional[str]): Device to load the model on.
            use_probabilities (bool): Whether to return probabilities instead
            of raw scores.
        """
        super().__init__(device)

        # Load model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer(
            "gpt2-helpful", device=self.device)
        self.use_probabilities = use_probabilities

        logger.info("GPT2 helpfulness reward model initialized")

    def get_rewards(self,
                    texts: List[str],
                    prompts: Optional[List[str]] = None) -> List[float]:
        """Calculate helpfulness rewards for the given texts.

        Args:
            texts (List[str]): List of text sequences to evaluate
            prompts (Optional[List[str]]): Prompt texts (required for this
            model)

        Returns:
            List[float]: Reward scores for each text

        Raises:
            ValueError: If prompts are not provided
        """
        if prompts is None or len(prompts) != len(texts):
            raise ValueError("Prompts must be provided and match the number "
                             "of texts for the GPT2 helpful model")

        # Extract continuations from texts
        continuations = [
            t[len(p):].strip() if t.startswith(p) else t
            for p, t in zip(prompts, texts)
        ]

        # Calculate helpfulness probabilities and scores
        probabilities, raw_scores = cal_gpt2_helpful_probabilities(
            prompts, continuations, self.model, self.tokenizer)

        # Return either probabilities or raw scores
        if self.use_probabilities:
            return probabilities
        else:
            return raw_scores
