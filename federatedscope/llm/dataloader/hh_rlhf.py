import random
import datasets
from federatedscope.core.auxiliaries.logging import logger
from federatedscope.llm.dataset.llm_dataset import LLMDataset

# The prompt dictionary remains the same
PROMPT_DICT = {
    "hh_cmp": (
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
    parts = text.strip().split('\n\n')
    if len(parts) < 2 or 'Assistant:' not in parts[-1]:
        return None, None
    response = parts[-1].replace('Assistant: ', '').strip()
    prompt = '\n\n'.join(parts[:-1]).strip()
    return prompt, response

def load_hh_rlhf_dataset(config, tokenizer):
    """
    Loads and processes the hh-rlhf dataset from Hugging Face.
    This function now returns a single unified dataset, which the framework
    will split according to the configuration.
    """
    logger.info("Loading and processing hh-rlhf dataset from Hugging Face...")

    try:
        # Load both subsets (this may take time on first run due to downloading)
        logger.info("Loading harmless-base dataset...")
        harmless_raw = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
        logger.info("Loading helpful-base dataset...")
        helpful_raw = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e

    def filter_fn(example):
        """Checks if the example can be parsed correctly."""
        prompt, chosen = parse_dialogue(example['chosen'])
        _, rejected = parse_dialogue(example['rejected'])
        return prompt is not None and chosen is not None and rejected is not None

    def preprocess(example):
        """Preprocesses a single example for choice-based training."""
        prompt, chosen = parse_dialogue(example['chosen'])
        _, rejected = parse_dialogue(example['rejected'])
        
        # Randomly swap to avoid positional bias
        if random.random() > 0.5:
            return {
                "prompt": prompt,
                "output_A": rejected,
                "output_B": chosen,
                "choice": " B"
            }
        else:
            return {
                "prompt": prompt,
                "output_A": chosen,
                "output_B": rejected,
                "choice": " A"
            }

    # Process all splits with parallel processing for faster execution
    # num_proc uses multiple CPU cores to speed up processing
    logger.info("Processing harmless-base train split...")
    harmless_train = harmless_raw['train'].filter(filter_fn, num_proc=4).map(preprocess, num_proc=4)
    logger.info("Processing harmless-base test split...")
    harmless_test = harmless_raw['test'].filter(filter_fn, num_proc=4).map(preprocess, num_proc=4)
    logger.info("Processing helpful-base train split...")
    helpful_train = helpful_raw['train'].filter(filter_fn, num_proc=4).map(preprocess, num_proc=4)
    logger.info("Processing helpful-base test split...")
    helpful_test = helpful_raw['test'].filter(filter_fn, num_proc=4).map(preprocess, num_proc=4)

    # Combine into a single training and test set
    logger.info("Combining datasets...")
    full_train_dataset = datasets.concatenate_datasets([harmless_train, helpful_train])
    full_test_dataset = datasets.concatenate_datasets([harmless_test, helpful_test])
    logger.info(f"Combined train dataset size: {len(full_train_dataset)}, test dataset size: {len(full_test_dataset)}")
    
    # Wrap the raw data into LLMDataset objects
    logger.info("Wrapping datasets into LLMDataset objects...")
    train_dataset = LLMDataset(full_train_dataset,
                               tokenizer,
                               prompt_input=PROMPT_DICT['hh_cmp'],
                               prompt_no_input=PROMPT_DICT['hh_cmp'],
                               output_tag='choice')
    
    test_dataset = LLMDataset(full_test_dataset,
                              tokenizer,
                              prompt_input=PROMPT_DICT['hh_cmp'],
                              prompt_no_input=PROMPT_DICT['hh_cmp'],
                              output_tag='choice')
    logger.info("Dataset loading completed!")

    # Return a tuple, just like reddit_tldr.py and shp.py
    # The framework will handle splitting this into train/val/test and
    # distributing it to clients.
    dataset = (train_dataset, test_dataset, test_dataset)

    return dataset, config


# Prompt dictionary for RLHF (generation and comparison)
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
