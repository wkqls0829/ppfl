"""Functions for calculating various types of reward scores."""

import logging
import nltk
from typing import List, Tuple, Any
import torch
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

try:
    # Try to initialize nltk requirements
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download nltk punkt: {e}")


def cal_gpt2_harmless_probabilities(
    prompts: List[str],
    continuations: List[str],
    model: Any,
    tokenizer: Any
) -> Tuple[List[float], List[float]]:
    """Calculate probabilities and raw scores indicating the harmlessness of a
    response.

    Args:
        prompts (List[str]): List of initial prompt strings
        continuations (List[str]): List of continuation (response) strings
        model: Model for evaluating harmlessness
        tokenizer: Tokenizer for processing the inputs

    Returns:
        Tuple[List[float], List[float]]: Harmlessness probabilities and raw
        scores
    """
    # Format inputs for model
    questions = ["\n\nHuman: " + p + " \n\nAssistant:" for p in prompts]
    probabilities = []
    raw_scores = []

    with torch.no_grad():
        for q, a in zip(questions, continuations):
            inputs = tokenizer(q,
                               a,
                               return_tensors='pt',
                               truncation=True).to(model.device)
            outputs = model(**inputs)

            prob = torch.sigmoid(outputs.logits).squeeze().item()
            probabilities.append(prob)

            raw_score = outputs.logits.squeeze().item()
            raw_scores.append(raw_score)

    return probabilities, raw_scores


def cal_gpt2_helpful_probabilities(
    prompts: List[str],
    continuations: List[str],
    model: Any,
    tokenizer: Any
) -> Tuple[List[float], List[float]]:
    """Calculate probabilities and raw scores indicating the helpfulness of a
    response.

    Args:
        prompts (List[str]): List of prompt strings
        continuations (List[str]): List of continuation (response) strings
        model: Model for evaluating helpfulness
        tokenizer: Tokenizer for processing the inputs

    Returns:
        Tuple[List[float], List[float]]: Helpfulness probabilities and raw
        scores
    """
    questions = ["\n\nHuman: " + p + " \n\nAssistant:" for p in prompts]
    probabilities = []
    raw_scores = []

    with torch.no_grad():
        for q, a in zip(questions, continuations):
            inputs = tokenizer(q,
                               a,
                               return_tensors='pt',
                               truncation=True).to(model.device)
            outputs = model(**inputs)

            prob = torch.sigmoid(outputs.logits).squeeze().item()
            probabilities.append(prob)

            raw_score = outputs.logits.squeeze().item()
            raw_scores.append(raw_score)

    return probabilities, raw_scores
