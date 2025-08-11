import logging
import datasets
import os
from federatedscope.core.data import StandaloneDataDict
from federatedscope.register import register_data

logger = logging.getLogger(__name__)

def parse_dialogue(text):
    """
    Parses the dialogue to separate the prompt from the response.
    """
    parts = text.strip().split("\n\nAssistant: ")
    if len(parts) > 1:
        prompt = "\n\nAssistant: ".join(parts[:-1]) + "\n\nAssistant: "
        response = parts[-1]
        return prompt, response
    return None, None

def _load_and_process_subset(base_dir):
    """
    Loads train and test splits from a specific base directory (e.g., 'harmless-base').

    Args:
        base_dir (str): The subdirectory within 'data/hh-rlhf' to load from.
    """
    # --- THIS IS THE KEY CHANGE ---
    # Construct the full, absolute paths to the data files inside their subdirectories
    data_path = os.path.join("data/hh-rlhf", base_dir)
    
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found at '{data_path}'. "
            f"Please ensure you have cloned the dataset correctly."
        )
        
    # The `datasets` library can automatically find train/test splits
    # and handle the .gz compression.
    dataset = datasets.load_dataset(data_path)
    # --- END OF CHANGE ---

    def preprocess(example):
        prompt_chosen, chosen_response = parse_dialogue(example['chosen'])
        prompt_rejected, rejected_response = parse_dialogue(example['rejected'])
        return {
            "prompt": prompt_chosen,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response
        }
    
    # Process both train and test splits
    processed_train = dataset['train'].map(preprocess, batched=False, num_proc=4).filter(lambda x: x['prompt'] is not None)
    processed_test = dataset['test'].map(preprocess, batched=False, num_proc=4).filter(lambda x: x['prompt'] is not None)
    
    return processed_train, processed_test


def load_hh_rlhf_data(config, client_cfgs=None):
    """
    Loads the hh-rlhf dataset and distributes it heterogeneously.
    """
    client_num = config.federate.client_num
    if client_num % 2 != 0:
        logger.warning(f"Client number ({client_num}) is odd. One group will have an extra client.")

    # Define the subdirectories for each dataset type
    harmless_dir = 'harmless-base'
    helpful_dir = 'helpful-base'
    
    logger.info(f"Loading and processing data from '{harmless_dir}'...")
    harmless_train_data, harmless_test_data = _load_and_process_subset(harmless_dir)

    logger.info(f"Loading and processing data from '{helpful_dir}'...")
    helpful_train_data, helpful_test_data = _load_and_process_subset(helpful_dir)
    
    data_dict = {}
    harmless_clients_num = client_num // 2
    
    logger.info(f"Assigning harmlessness data to {harmless_clients_num} clients...")
    if harmless_clients_num > 0:
        for i in range(harmless_clients_num):
            client_id = i + 1
            train_shard = harmless_train_data.shard(num_shards=harmless_clients_num, index=i)
            test_shard = harmless_test_data.shard(num_shards=harmless_clients_num, index=i)
            data_dict[client_id] = {'train': train_shard, 'test': test_shard, 'val': test_shard}

    helpful_clients_num = client_num - harmless_clients_num
    logger.info(f"Assigning helpfulness data to {helpful_clients_num} clients...")
    if helpful_clients_num > 0:
        for i in range(helpful_clients_num):
            client_id = i + harmless_clients_num + 1
            train_shard = helpful_train_data.shard(num_shards=helpful_clients_num, index=i)
            test_shard = helpful_test_data.shard(num_shards=helpful_clients_num, index=i)
            data_dict[client_id] = {'train': train_shard, 'test': test_shard, 'val': test_shard}

    logger.info("Finished creating heterogeneous data distribution.")
    return StandaloneDataDict(data_dict, config), config

def call_hh_rlhf(config, client_cfgs):
    if config.data.type == "hh-rlhf":
        data, modified_config = load_hh_rlhf_data(config, client_cfgs)
        return data, modified_config

register_data("hh-rlhf", call_hh_rlhf)
