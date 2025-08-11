import logging
import datasets
from federatedscope.core.data import StandaloneDataDict
from federatedscope.register import register_data

logger = logging.getLogger(__name__)

def parse_dialogue(text):
    """
    Parses the dialogue to separate the prompt from the response.
    The prompt is everything before the last "Assistant:" turn.
    """
    parts = text.strip().split("\n\nAssistant: ")
    if len(parts) > 1:
        prompt = "\n\nAssistant: ".join(parts[:-1]) + "\n\nAssistant: "
        response = parts[-1]
        return prompt, response
    return None, None

def _load_and_process_subset(data_files):
    """Loads a specific subset of data files and processes them."""
    dataset = datasets.load_dataset("Anthropic/hh-rlhf", data_files=data_files)
    
    def preprocess(example):
        prompt_chosen, chosen_response = parse_dialogue(example['chosen'])
        prompt_rejected, rejected_response = parse_dialogue(example['rejected'])
        return {
            "prompt": prompt_chosen,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response
        }
        
    processed_dataset = dataset.map(preprocess, batched=False, num_proc=4)
    processed_dataset = processed_dataset.filter(lambda x: x['prompt'] is not None and x['chosen_response'] is not None and x['rejected_response'] is not None)
    return processed_dataset['train'] # The loaded data is all in the 'train' split

def load_hh_rlhf_data(config, client_cfgs=None):
    """
    Loads the hh-rlhf dataset and distributes it heterogeneously.
    - Half clients get 'harmlessness' data.
    - The other half get 'helpfulness' data.
    """
    client_num = config.federate.client_num
    if client_num % 2 != 0:
        logger.warning(f"Client number ({client_num}) is odd. One group will have an extra client.")

    # Load harmlessness data
    logger.info("Loading harmlessness dataset...")
    harmless_files = {'train': ['harmlessness-base-train.jsonl'], 'test': ['harmlessness-base-test.jsonl']}
    harmless_train_data = _load_and_process_subset(harmless_files['train'])
    harmless_test_data = _load_and_process_subset(harmless_files['test'])

    # Load helpfulness data
    logger.info("Loading helpfulness dataset...")
    helpful_files = {'train': ['helpfulness-base-train.jsonl'], 'test': ['helpfulness-base-test.jsonl']}
    helpful_train_data = _load_and_process_subset(helpful_files['train'])
    helpful_test_data = _load_and_process_subset(helpful_files['test'])
    
    # Manually create the data dictionary for clients
    data_dict = {}
    
    # Assign data to clients
    harmless_clients_num = client_num // 2
    
    # Distribute harmlessness data to the first half of clients
    logger.info(f"Assigning harmlessness data to {harmless_clients_num} clients...")
    harmless_train_shards = harmless_train_data.shard(num_shards=harmless_clients_num, index=0, contiguous=True)
    harmless_test_shards = harmless_test_data.shard(num_shards=harmless_clients_num, index=0, contiguous=True)
    for i in range(harmless_clients_num):
        client_id = i + 1
        train_shard = harmless_train_shards.shard(num_shards=harmless_clients_num, index=i)
        test_shard = harmless_test_shards.shard(num_shards=harmless_clients_num, index=i)
        data_dict[client_id] = {
            'train': train_shard,
            'test': test_shard,
            'val': test_shard.select(range(len(test_shard) // 2)) # Use half of test for validation
        }

    # Distribute helpfulness data to the second half of clients
    helpful_clients_num = client_num - harmless_clients_num
    logger.info(f"Assigning helpfulness data to {helpful_clients_num} clients...")
    helpful_train_shards = helpful_train_data.shard(num_shards=helpful_clients_num, index=0, contiguous=True)
    helpful_test_shards = helpful_test_data.shard(num_shards=helpful_clients_num, index=0, contiguous=True)
    for i in range(helpful_clients_num):
        client_id = i + harmless_clients_num + 1
        train_shard = helpful_train_shards.shard(num_shards=helpful_clients_num, index=i)
        test_shard = helpful_test_shards.shard(num_shards=helpful_clients_num, index=i)
        data_dict[client_id] = {
            'train': train_shard,
            'test': test_shard,
            'val': test_shard.select(range(len(test_shard) // 2)) # Use half of test for validation
        }

    logger.info("Finished creating heterogeneous data distribution.")
    return StandaloneDataDict(data_dict, config), config

def call_hh_rlhf(config, client_cfgs):
    if config.data.type == "hh-rlhf":
        data, modified_config = load_hh_rlhf_data(config, client_cfgs)
        return data, modified_config

register_data("hh-rlhf", call_hh_rlhf)
