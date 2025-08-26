import datasets
from federatedscope.core.auxiliaries.logging import logger
from federatedscope.llm.dataset.llm_dataset import LLMDataset

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

def parse_dialogue(text):
    """Helper to split dialogue into prompt and the final assistant response."""
    parts = text.split('\n\n')
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
        # Load both subsets
        harmless_raw = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
        helpful_raw = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e

    def preprocess(example):
        """Preprocesses a single example for choice-based training."""
        prompt, chosen = parse_dialogue(example['chosen'])
        _, rejected = parse_dialogue(example['rejected'])

        if prompt is None or chosen is None or rejected is None:
            return None
        
        return {
            "prompt": prompt,
            "output_A": chosen,
            "output_B": rejected,
            "choice": " A"  # Target for the choice trainer
        }

    # Process all splits
    harmless_train = harmless_raw['train'].map(preprocess).filter(lambda x: x is not None)
    harmless_test = harmless_raw['test'].map(preprocess).filter(lambda x: x is not None)
    helpful_train = helpful_raw['train'].map(preprocess).filter(lambda x: x is not None)
    helpful_test = helpful_raw['test'].map(preprocess).filter(lambda x: x is not None)

    # Combine into a single training and test set
    full_train_dataset = datasets.concatenate_datasets([harmless_train, helpful_train])
    full_test_dataset = datasets.concatenate_datasets([harmless_test, helpful_test])
    
    # Wrap the raw data into LLMDataset objects
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

    # Return a tuple, just like reddit_tldr.py and shp.py
    # The framework will handle splitting this into train/val/test and
    # distributing it to clients.
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
    list_prompts = list_prompts.to_dict()['prompt']

    if raw_no_prompt:
        if max_num_test > 0:
            return (list_prompts[:max_num_test], None, None)
        else:
            return (list_prompts, None, None)

    # This part is for federated training, not standalone, but we keep the
    # structure for consistency. It won't be used by standalone_training.py
    # when `raw_no_prompt` is True.
    return ([], [], [])
