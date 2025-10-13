import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import datasets

from federatedscope.core.auxiliaries.logging import logger
from federatedscope.llm.dataset.llm_dataset import LLMDataset, \
    LLMComparisonDataset


HH_RLHF_PROMPT_DICT = {
    "generation": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is both helpful and harmless.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "comparison": (
        "Below is a conversation between a human and an AI assistant, "
        "followed by two responses. Pick the response that is more "
        "helpful and harmless. State your choice with a single capital "
        "letter, i.e., \"A\" if RESPONSE A is better, "
        "\"B\" if RESPONSE B is better.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE A: {output_A}\n"
        "### RESPONSE B: {output_B}\n"
        "### YOUR CHOICE:"
    ),
}


def parse_dialogue(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a dialogue string into prompt context and final assistant reply."""

    if not text:
        return None, None

    if "Assistant:" not in text:
        return None, None

    prompt_part, assistant_part = text.rsplit("Assistant:", 1)
    prompt_part = prompt_part.strip()
    assistant_part = assistant_part.strip()

    if not prompt_part or not assistant_part:
        return None, None

    return prompt_part, assistant_part


def _load_split(data_root: str,
                subset: str,
                split: str) -> List[Dict[str, str]]:
    """Load and preprocess a split from the Anthropic HH-RLHF dataset."""

    os.makedirs(data_root, exist_ok=True)
    logger.info("Loading hh-rlhf subset %s/%s", subset, split)

    try:
        raw_split = datasets.load_dataset("Anthropic/hh-rlhf",
                                          data_dir=subset,
                                          split=split,
                                          cache_dir=data_root)
    except Exception as error:  # pragma: no cover - network failure
        logger.error("Failed to load %s/%s from Hugging Face: %s", subset,
                     split, error)
        raise

    processed_examples: List[Dict[str, str]] = []
    skipped = 0

    for example in raw_split:
        prompt_chosen, chosen = parse_dialogue(example.get("chosen", ""))
        prompt_rejected, rejected = parse_dialogue(example.get(
            "rejected", ""))

        prompt = prompt_chosen or prompt_rejected

        if prompt is None or chosen is None or rejected is None:
            skipped += 1
            continue

        if prompt_rejected and prompt_chosen and prompt_rejected.strip(
        ) != prompt_chosen.strip():
            logger.debug("Mismatched prompts detected in %s/%s", subset, split)

        processed_examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "category": subset,
        })

    if skipped:
        logger.warning("Skipped %d malformed dialogues from %s/%s", skipped,
                       subset, split)

    return processed_examples


def _build_splits(data_root: str,
                  seed: int = 42) -> Tuple[List[Dict[str, str]],
                                           List[Dict[str, str]],
                                           List[Dict[str, str]]]:
    """Create train/val/test splits from the HH-RLHF dataset."""

    harmless_train = _load_split(data_root, "harmless-base", "train")
    helpful_train = _load_split(data_root, "helpful-base", "train")

    harmless_test = _load_split(data_root, "harmless-base", "test")
    helpful_test = _load_split(data_root, "helpful-base", "test")

    list_train_dict = harmless_train + helpful_train
    list_eval_dict = harmless_test + helpful_test

    random.Random(seed).shuffle(list_eval_dict)
    if not list_eval_dict:
        return list_train_dict, [], []

    split_point = max(1, len(list_eval_dict) // 2)
    list_val_dict = list_eval_dict[:split_point]
    list_test_dict = list_eval_dict[split_point:] or list_eval_dict[:split_point]

    return list_train_dict, list_val_dict, list_test_dict


def _choice_view(list_data_dict: Sequence[Dict[str, str]],
                 augment: bool = False) -> List[Dict[str, str]]:
    """Create choice-format samples for LLMDataset consumption."""

    formatted_list: List[Dict[str, str]] = []
    for sample in list_data_dict:
        base_sample = {
            "prompt": sample["prompt"],
            "output_A": sample["chosen"],
            "output_B": sample["rejected"],
            "choice": " A",
            "category": sample.get("category"),
        }
        formatted_list.append(base_sample)

        if augment:
            formatted_list.append({
                "prompt": sample["prompt"],
                "output_A": sample["rejected"],
                "output_B": sample["chosen"],
                "choice": " B",
                "category": sample.get("category"),
            })

    return formatted_list


def _comparison_view(list_data_dict: Sequence[Dict[str, str]]) -> List[Dict[
        str, str]]:
    """Create comparison-format samples for LLMComparisonDataset."""

    comparison_list: List[Dict[str, str]] = []
    for sample in list_data_dict:
        comparison_list.append({
            "prompt": sample["prompt"],
            "output_A": sample["rejected"],
            "output_B": sample["chosen"],
            "choice": 1,
            "category": sample.get("category"),
        })

    return comparison_list


def load_hh_rlhf_dataset(config, tokenizer):
    """Return HH-RLHF data in the same structure as other LLM loaders."""

    data_root = os.path.join(config.data.root, "hh-rlhf")
    logger.info("Preparing hh-rlhf dataset using TL;DR-style preprocessing...")

    list_train_dict, list_val_dict, list_test_dict = _build_splits(data_root)

    train_dataset = LLMDataset(_choice_view(list_train_dict, augment=True),
                               tokenizer,
                               prompt_input=HH_RLHF_PROMPT_DICT["comparison"],
                               prompt_no_input=HH_RLHF_PROMPT_DICT["comparison"],
                               output_tag="choice")
    val_dataset = LLMDataset(_choice_view(list_val_dict),
                             tokenizer,
                             prompt_input=HH_RLHF_PROMPT_DICT["comparison"],
                             prompt_no_input=HH_RLHF_PROMPT_DICT["comparison"],
                             output_tag="choice")
    test_dataset = LLMDataset(_choice_view(list_test_dict),
                              tokenizer,
                              prompt_input=HH_RLHF_PROMPT_DICT["comparison"],
                              prompt_no_input=HH_RLHF_PROMPT_DICT["comparison"],
                              output_tag="choice")

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset, config


def load_comparison_dataset(data_root: str,
                            tokenizer,
                            max_num_test: int = -1):
    """Load HH-RLHF pairwise comparisons for reward modelling."""

    list_train_dict, list_val_dict, list_test_dict = _build_splits(data_root)

    train_dataset = LLMComparisonDataset(_comparison_view(list_train_dict),
                                         tokenizer,
                                         prompt_input=HH_RLHF_PROMPT_DICT[
                                             "comparison"],
                                         prompt_no_input=HH_RLHF_PROMPT_DICT[
                                             "comparison"],
                                         output_A="output_A",
                                         output_B="output_B",
                                         choice="choice")
    val_dataset = LLMComparisonDataset(_comparison_view(list_val_dict),
                                       tokenizer,
                                       prompt_input=HH_RLHF_PROMPT_DICT[
                                           "comparison"],
                                       prompt_no_input=HH_RLHF_PROMPT_DICT[
                                           "comparison"],
                                       output_A="output_A",
                                       output_B="output_B",
                                       choice="choice")
    test_dataset = LLMComparisonDataset(_comparison_view(list_test_dict),
                                        tokenizer,
                                        prompt_input=HH_RLHF_PROMPT_DICT[
                                            "comparison"],
                                        prompt_no_input=HH_RLHF_PROMPT_DICT[
                                            "comparison"],
                                        output_A="output_A",
                                        output_B="output_B",
                                        choice="choice")

    if max_num_test > 0:
        val_dataset.win_dataset.input_ids = val_dataset.win_dataset.input_ids[:
            max_num_test]
        val_dataset.lose_dataset.input_ids = val_dataset.lose_dataset.input_ids[:
            max_num_test]
        test_dataset.win_dataset.input_ids = test_dataset.win_dataset.input_ids[:
            max_num_test]
        test_dataset.lose_dataset.input_ids = test_dataset.lose_dataset.input_ids[:
            max_num_test]

    return train_dataset, val_dataset, test_dataset


def load_hh_rlhf_for_rlhf(data_root,
                          config,
                          max_num_test: int = -1,
                          raw_no_prompt: bool = False):
    """Provide prompts for the standalone RLHF pipeline."""

    logger.info("Loading hh-rlhf prompts from Hugging Face for RLHF...")

    prompt_list: List[Dict[str, str]] = []
    for subset in ("harmless-base", "helpful-base"):
        split_examples = _load_split(data_root, subset, "test")
        for sample in split_examples:
            prompt_list.append({"prompt": sample["prompt"], "category": subset})

    if max_num_test > 0:
        prompt_list = prompt_list[:max_num_test]

    if raw_no_prompt:
        return prompt_list, None, None

    return [], [], []
