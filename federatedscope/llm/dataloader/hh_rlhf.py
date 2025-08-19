import os
import datasets
from federatedscope.core.data import StandaloneDataDict, ClientData
from federatedscope.core.auxiliaries.logging import logger
from federatedscope.llm.dataset.llm_dataset import LLMDataset

# This prompt dictionary now uses keys that we will create in our
# preprocess function. The key is to use generic names like 'prompt'
# instead of specific ones like 'instruction'.
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
    parts = text.split('\n\n')
    if len(parts) < 2 or 'Assistant:' not in parts[-1]:
        return None, None
    response = parts[-1].replace('Assistant: ', '').strip()
    prompt = '\n\n'.join(parts[:-1]).strip()
    return prompt, response

def load_hh_rlhf_dataset(config, tokenizer):
    """
    Loads, processes, and distributes the hh-rlhf dataset from Hugging Face
    for choice-based training.
    """
    logger.info("Loading hh-rlhf dataset from Hugging Face for choice task...")

    try:
        harmless_raw = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
        helpful_raw = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e

    def preprocess(example):
        """
        Preprocesses a single example to create the fields our prompt expects:
        'prompt', 'output_A', 'output_B', and 'choice'.
        """
        prompt, chosen = parse_dialogue(example['chosen'])
        _, rejected = parse_dialogue(example['rejected'])

        if prompt is None or chosen is None or rejected is None:
            return None
        
        return {
            "prompt": prompt,
            "output_A": chosen,
            "output_B": rejected,
            "choice": " A"  # The target label for the trainer
        }

    # Process and filter out failed examples
    harmless_train = harmless_raw['train'].map(preprocess).filter(lambda x: x is not None)
    harmless_test = harmless_raw['test'].map(preprocess).filter(lambda x: x is not None)
    helpful_train = helpful_raw['train'].map(preprocess).filter(lambda x: x is not None)
    helpful_test = helpful_raw['test'].map(preprocess).filter(lambda x: x is not None)
    
    # --- Create Federated Data Split ---
    client_num = config.federate.client_num
    data_dict = {}
    
    harmless_clients_num = client_num // 2
    helpful_clients_num = client_num - harmless_clients_num

    logger.info(f"Assigning 'harmless' data to {harmless_clients_num} clients...")
    for i in range(harmless_clients_num):
        client_id = i + 1
        train_shard = harmless_train.shard(harmless_clients_num, i)
        test_shard = harmless_test.shard(harmless_clients_num, i)
        data_dict[client_id] = ClientData(
            client_cfg=config,
            train_data=LLMDataset(train_shard, tokenizer, 
                                  prompt_input=PROMPT_DICT['hh_cmp'],
                                  prompt_no_input=PROMPT_DICT['hh_cmp'], 
                                  output_tag='choice'),
            val_data=LLMDataset(test_shard, tokenizer, 
                                prompt_input=PROMPT_DICT['hh_cmp'],
                                prompt_no_input=PROMPT_DICT['hh_cmp'],
                                output_tag='choice'),
            test_data=LLMDataset(test_shard, tokenizer, 
                                 prompt_input=PROMPT_DICT['hh_cmp'],
                                 prompt_no_input=PROMPT_DICT['hh_cmp'],
                                 output_tag='choice')
        )

    logger.info(f"Assigning 'helpful' data to {helpful_clients_num} clients...")
    for i in range(helpful_clients_num):
        client_id = i + harmless_clients_num + 1
        train_shard = helpful_train.shard(helpful_clients_num, i)
        test_shard = helpful_test.shard(helpful_clients_num, i)
        data_dict[client_id] = ClientData(
            client_cfg=config,
            train_data=LLMDataset(train_shard, tokenizer, 
                                  prompt_input=PROMPT_DICT['hh_cmp'],
                                  prompt_no_input=PROMPT_DICT['hh_cmp'], 
                                  output_tag='choice'),
            val_data=LLMDataset(test_shard, tokenizer, 
                                prompt_input=PROMPT_DICT['hh_cmp'],
                                prompt_no_input=PROMPT_DICT['hh_cmp'],
                                output_tag='choice'),
            test_data=LLMDataset(test_shard, tokenizer, 
                                 prompt_input=PROMPT_DICT['hh_cmp'],
                                 prompt_no_input=PROMPT_DICT['hh_cmp'],
                                 output_tag='choice')
        )
        
    # --- Create Global Dataset for Server ---
    global_train = datasets.concatenate_datasets([harmless_train, helpful_train])
    global_test = datasets.concatenate_datasets([harmless_test, helpful_test])

    data_container = StandaloneDataDict(data_dict, config)
    data_container.train_data = LLMDataset(global_train, tokenizer, 
                                          prompt_input=PROMPT_DICT['hh_cmp'],
                                          prompt_no_input=PROMPT_DICT['hh_cmp'],
                                          output_tag='choice')
    data_container.test_data = LLMDataset(global_test, tokenizer, 
                                         prompt_input=PROMPT_DICT['hh_cmp'],
                                         prompt_no_input=PROMPT_DICT['hh_cmp'],
                                         output_tag='choice')
    data_container.val_data = data_container.test_data

    return data_container, config
