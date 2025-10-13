import os
import random

import datasets
from federatedscope.core.auxiliaries.logging import logger
from federatedscope.llm.dataset.llm_dataset import LLMComparisonDataset


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
    )
}

def parse_dialogue(text):
    """Helper to split dialogue into prompt and the final assistant response."""
    if "Assistant:" not in text:
        return None, None

def _collect_split(data_dir, split):
    """Download and preprocess a single split from the Anthropic HH-RLHF hub."""

    try:
        raw_split = datasets.load_dataset("Anthropic/hh-rlhf",
                                          data_dir=data_dir,
                                          split=split)
    except Exception as error:
        logger.error("Failed to load %s/%s from Hugging Face: %s", data_dir,
                     split, error)
        raise

    processed_examples = []
    for example in raw_split:
        prompt, chosen = parse_dialogue(example.get('chosen', ''))
        _, rejected = parse_dialogue(example.get('rejected', ''))

        if prompt is None or chosen is None or rejected is None:
            continue

        processed_examples.append({
            "prompt": prompt,
            "output_A": chosen,
            "output_B": rejected,
            "choice": " A",
        })

    return processed_examples


def load_hh_rlhf_dataset(config, tokenizer):
    """Return HH-RLHF data following the same structure as other LLM loaders."""

    logger.info("Preparing hh-rlhf dataset using TL;DR-style preprocessing...")

    harmless_train = _collect_split("harmless-base", "train")
    harmless_test = _collect_split("harmless-base", "test")
    helpful_train = _collect_split("helpful-base", "train")
    helpful_test = _collect_split("helpful-base", "test")

    list_train_dict = harmless_train + helpful_train
    list_eval_dict = harmless_test + helpful_test

    random.Random(42).shuffle(list_eval_dict)
    split_point = len(list_eval_dict) // 2
    list_val_dict = list_eval_dict[:split_point]
    list_test_dict = list_eval_dict[split_point:]

    if not list_test_dict:
        list_test_dict = list_val_dict

    train_dataset = LLMDataset(list_train_dict,
                               tokenizer,
                               prompt_input=HH_RLHF_PROMPT_DICT['comparison'],
                               prompt_no_input=HH_RLHF_PROMPT_DICT['comparison'],
                               output_tag='choice')
    val_dataset = LLMDataset(list_val_dict,
                             tokenizer,
                             prompt_input=HH_RLHF_PROMPT_DICT['comparison'],
                             prompt_no_input=HH_RLHF_PROMPT_DICT['comparison'],
                             output_tag='choice')
    test_dataset = LLMDataset(list_test_dict,
                              tokenizer,
                              prompt_input=HH_RLHF_PROMPT_DICT['comparison'],
                              prompt_no_input=HH_RLHF_PROMPT_DICT['comparison'],
                              output_tag='choice')

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset, config

def load_hh_rlhf_for_rlhf(data_root,
                          config,
                          max_num_test=-1,
                          raw_no_prompt=False):
    """
    Loads and processes the hh-rlhf dataset from Hugging Face for the
    standalone RLHF script. It combines both helpful and harmless datasets.
    """
    logger.info("Loading hh-rlhf prompts from Hugging Face for RLHF...")

    # Load both "harmless" and "helpful" test sets for prompts
    try:
        harmless_test = datasets.load_dataset("Anthropic/hh-rlhf",
                                              data_dir="harmless-base",
                                              split='test')
        helpful_test = datasets.load_dataset("Anthropic/hh-rlhf",
                                             data_dir="helpful-base",
                                             split='test')
    except Exception as e:
        logger.error(
            f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e

    # Combine them for a diverse set of prompts
    combined_prompts_dataset = datasets.concatenate_datasets(
        [harmless_test, helpful_test])

    def get_prompt(example):
        # The prompt is the same for 'chosen' and 'rejected'
        prompt, _ = parse_dialogue(example['chosen'])
        if prompt:
            return {'prompt': prompt}
        else:
            # Return a key with a None value to allow for filtering
            return {'prompt': None}

    # Extract all prompts and filter out any that failed parsing
    list_prompts = combined_prompts_dataset.map(get_prompt).filter(
        lambda x: x['prompt'] is not None
    )

    # Convert to the simple list of dictionaries format
    list_prompts = list(list_prompts)

    if raw_no_prompt:
        if max_num_test > 0:
            return (list_prompts[:max_num_test], None, None)
        else:
            return (list_prompts, None, None)

    # This part is for federated training, not standalone, but we keep the
    # structure for consistency. It won't be used by standalone_training.py
    # when `raw_no_prompt` is True.
    return ([], [], [])
