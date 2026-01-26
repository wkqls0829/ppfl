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
    "generation_helpful": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is helpful.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "generation_harmless": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is harmless.\n\n"
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

    # Limit dataset size for testing (read from config, default: -1 = no limit)
    # If config.data.max_train_samples or config.data.max_test_samples is set, use it
    # Otherwise, use full dataset (default: -1 = no limit)
    max_train_samples = getattr(config.data, 'max_train_samples', -1)  # Default: -1 (no limit)
    max_test_samples = getattr(config.data, 'max_test_samples', -1)  # Default: -1 (no limit)
    
    if max_train_samples > 0:
        # Limit each subset proportionally
        harmless_train_size = min(len(harmless_train), max_train_samples // 2)
        helpful_train_size = min(len(helpful_train), max_train_samples // 2)
        harmless_train = harmless_train.select(range(harmless_train_size))
        helpful_train = helpful_train.select(range(helpful_train_size))
        logger.info(f"Limited train dataset to {harmless_train_size + helpful_train_size} samples (max_train_samples={max_train_samples})")
    
    if max_test_samples > 0:
        # Limit each subset proportionally
        harmless_test_size = min(len(harmless_test), max_test_samples // 2)
        helpful_test_size = min(len(helpful_test), max_test_samples // 2)
        harmless_test = harmless_test.select(range(harmless_test_size))
        helpful_test = helpful_test.select(range(helpful_test_size))
        logger.info(f"Limited test dataset to {harmless_test_size + helpful_test_size} samples (max_test_samples={max_test_samples})")

    # Combine into a single training and test set
    full_train_dataset = datasets.concatenate_datasets([harmless_train, helpful_train])
    full_test_dataset = datasets.concatenate_datasets([harmless_test, helpful_test])
    
    # Wrap the raw data into LLMDataset objects
    train_dataset = LLMDataset(full_train_dataset,
                               tokenizer,
                               prompt_input=HH_RLHF_PROMPT_DICT['comparison'],
                               prompt_no_input=HH_RLHF_PROMPT_DICT['comparison'],
                               output_tag='choice')
    
    test_dataset = LLMDataset(full_test_dataset,
                              tokenizer,
                              prompt_input=HH_RLHF_PROMPT_DICT['comparison'],
                              prompt_no_input=HH_RLHF_PROMPT_DICT['comparison'],
                              output_tag='choice')

    # Return a tuple, just like reddit_tldr.py and shp.py
    # The framework will handle splitting this into train/val/test and
    # distributing it to clients.
    dataset = (train_dataset, test_dataset, test_dataset)

    return dataset, config

def load_hh_rlhf_for_rlhf(data_root,
                          config,
                          max_num_test=-1,
                          raw_no_prompt=False,
                          split_by_client=False,
                          client_num=None):
    """
    Loads and processes the hh-rlhf dataset from Hugging Face for the
    standalone RLHF script. It combines both helpful and harmless datasets.
    
    Args:
        data_root: Root directory for data
        config: Configuration object
        max_num_test: Maximum number of test samples per client (if split_by_client=True) or total (if False)
        raw_no_prompt: If True, return raw prompts without processing
        split_by_client: If True, split test data by client (harmless: 1 to client_num//2, helpful: client_num//2+1 to client_num)
        client_num: Number of clients (required if split_by_client=True)
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

    def get_prompt(example):
        # The prompt is the same for 'chosen' and 'rejected'
        prompt, _ = parse_dialogue(example['chosen'])
        if prompt:
            return {'prompt': prompt}
        else:
            # Return a key with a None value to allow for filtering
            return {'prompt': None}

    # Extract prompts and filter out any that failed parsing
    harmless_prompts = harmless_test.map(get_prompt).filter(
        lambda x: x['prompt'] is not None
    )
    helpful_prompts = helpful_test.map(get_prompt).filter(
        lambda x: x['prompt'] is not None
    )

    if split_by_client and client_num is not None:
        # Split by client: harmless -> clients 1 to client_num//2, helpful -> clients client_num//2+1 to client_num
        harmless_clients_num = client_num // 2
        helpful_clients_num = client_num - harmless_clients_num
        
        # Convert to list format
        harmless_prompts_list = list(harmless_prompts)
        helpful_prompts_list = list(helpful_prompts)
        
        # Split harmless prompts by client
        harmless_per_client = len(harmless_prompts_list) // harmless_clients_num if harmless_clients_num > 0 else 0
        helpful_per_client = len(helpful_prompts_list) // helpful_clients_num if helpful_clients_num > 0 else 0
        
        # Create client-specific test data
        client_test_data = {}
        
        # Assign harmless data to clients 1 to harmless_clients_num
        for i in range(harmless_clients_num):
            client_id = i + 1
            start_idx = i * harmless_per_client
            end_idx = (i + 1) * harmless_per_client if i < harmless_clients_num - 1 else len(harmless_prompts_list)
            client_prompts = harmless_prompts_list[start_idx:end_idx]
            
            # Limit per client if max_num_test is specified
            if max_num_test > 0:
                client_prompts = client_prompts[:max_num_test]
            
            client_test_data[client_id] = client_prompts
        
        # Assign helpful data to clients harmless_clients_num+1 to client_num
        for i in range(helpful_clients_num):
            client_id = harmless_clients_num + i + 1
            start_idx = i * helpful_per_client
            end_idx = (i + 1) * helpful_per_client if i < helpful_clients_num - 1 else len(helpful_prompts_list)
            client_prompts = helpful_prompts_list[start_idx:end_idx]
            
            # Limit per client if max_num_test is specified
            if max_num_test > 0:
                client_prompts = client_prompts[:max_num_test]
            
            client_test_data[client_id] = client_prompts
        
        logger.info(f"Split test data by client: {len(client_test_data)} clients, "
                   f"harmless clients: 1-{harmless_clients_num}, helpful clients: {harmless_clients_num+1}-{client_num}")
        for client_id, prompts in client_test_data.items():
            logger.info(f"  Client {client_id}: {len(prompts)} test prompts")
        
        if raw_no_prompt:
            return (client_test_data, None, None)
        else:
            return (client_test_data, None, None)
    else:
        # Combine them for a diverse set of prompts (original behavior)
        combined_prompts_dataset = datasets.concatenate_datasets(
            [harmless_prompts, helpful_prompts])

        # Convert to the simple list of dictionaries format
        list_prompts = list(combined_prompts_dataset)

        if raw_no_prompt:
            if max_num_test > 0:
                return (list_prompts[:max_num_test], None, None)
            else:
                return (list_prompts, None, None)

        # This part is for federated training, not standalone, but we keep the
        # structure for consistency. It won't be used by standalone_training.py
        # when `raw_no_prompt` is True.
        return ([], [], [])
