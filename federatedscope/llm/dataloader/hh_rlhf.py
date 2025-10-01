import os

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
    parts = text.split('\n\n')
    if len(parts) < 2 or 'Assistant:' not in parts[-1]:
        return None, None
    response = parts[-1].replace('Assistant: ', '').strip()
    prompt = '\n\n'.join(parts[:-1]).strip()
    return prompt, response

def _build_comparison_list(hf_split, subset_tag):
    """Convert a Hugging Face split into LLM entries."""
    processed = []
    skipped = 0
    for example in hf_split:
        prompt, chosen = parse_dialogue(example['chosen'])
        _, rejected = parse_dialogue(example['rejected'])

        if prompt is None or chosen is None or rejected is None:
            skipped += 1
            continue
        processed.append({
            "prompt": prompt,
            "output_A": rejected,
            "output_B": chosen,
            "choice": 1,
            "category": subset_tag,
        })
    if skipped:
        logger.warning(
            "Skipped %d malformed hh-rlhf records from the %s split.",
            skipped,
            subset_tag,
        )

    return processed

def load_hh_rlhf_dataset(config, tokenizer):
    logger.info("Loading and processing hh-rlhf dataset from Hugging Face...")

    try:
        harmless_raw = datasets.load_dataset("Anthropic/hh-rlhf",
                                             data_dir="harmless-base")
        helpful_raw = datasets.load_dataset("Anthropic/hh-rlhf",
                                            data_dir="helpful-base")
    except Exception as e:
        logger.error("Failed to load dataset from Hugging Face. Error: %s", e)
        raise

    harmless_train = _build_comparison_list(harmless_raw['train'], 'harmless')
    harmless_test = _build_comparison_list(harmless_raw['test'], 'harmless')
    helpful_train = _build_comparison_list(helpful_raw['train'], 'helpful')
    helpful_test = _build_comparison_list(helpful_raw['test'], 'helpful')

    train_list = harmless_train + helpful_train
    test_list = harmless_test + helpful_test

    train_dataset = LLMComparisonDataset(
        train_list,
        tokenizer,
        prompt_input=HH_RLHF_PROMPT_DICT['generation'],
        prompt_no_input=HH_RLHF_PROMPT_DICT['generation'],
        output_A='output_A',
        output_B='output_B',
        choice='choice')

    test_dataset = LLMComparisonDataset(
        test_list,
        tokenizer,
        prompt_input=HH_RLHF_PROMPT_DICT['generation'],
        prompt_no_input=HH_RLHF_PROMPT_DICT['generation'],
        output_A='output_A',
        output_B='output_B',
        choice='choice')



    dataset = (train_dataset, test_dataset, test_dataset)

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
