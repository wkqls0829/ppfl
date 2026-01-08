import logging
import os
import random
import json
from tqdm import tqdm
import numpy as np
import gc
import copy
from itertools import combinations

import torch
from torch.utils.data import DataLoader

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.data import ClientData
from federatedscope.llm.dataloader import LLMDataCollator
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.dataset.llm_dataset import (
    DefaultToken,
    LLMDataset,
    LLMComparisonDataset,
)
from federatedscope.llm.trainer.reward_trainer import (
    DPORewardTrainer,
    _get_batch_logps,
    dpo_loss,
)
from federatedscope.core.auxiliaries.utils import add_prefix_to_path

logger = logging.getLogger(__name__)


@torch.no_grad()
def cal_acc(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    # print(new_logits)
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)

    return new_labels, new_logits, predicted, predicted.eq(
        new_labels).sum().item()

def _get_model_device(model, fallback="cpu"):
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "module"):
        device = getattr(model.module, "device", None)

    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = fallback

    if isinstance(device, torch.device):
        return device

    try:
        return torch.device(device)
    except (TypeError, ValueError):
        return torch.device(fallback)


def get_rlhf_prompts_dataset(config):
    dataset_name, _ = config.data.type.split("@")

    if dataset_name.lower() == "reddit-tldr-rlhf":
        from federatedscope.llm.dataloader.reddit_tldr import (
            load_human_finetuning_dataset,
            TLDR_PROMPT_DICT,
        )

        data_root = os.path.join(config.data.root, "reddit-tldr-comparison")
        list_train_prompts, _, _ = load_human_finetuning_dataset(
            data_root,
            tokenizer=None,
            rlhf=True,
            max_num_test=1000,
            raw_no_prompt=True)
        generation_prompt = TLDR_PROMPT_DICT["summary"]
        selector_prompt = TLDR_PROMPT_DICT["summary_cmp"]

    elif dataset_name.lower() == "hh-rlhf":
        from federatedscope.llm.dataloader.hh_rlhf import (
            load_hh_rlhf_for_rlhf,
            HH_RLHF_PROMPT_DICT,
        )
        data_root = os.path.join(config.data.root, "hh-rlhf")

        list_train_prompts, _, _ = load_hh_rlhf_for_rlhf(
            data_root,
            config,
            max_num_test=1000,
            raw_no_prompt=True,
        )

        generation_prompt = HH_RLHF_PROMPT_DICT["generation"]
        selector_prompt = HH_RLHF_PROMPT_DICT["comparison"]

    elif dataset_name.lower() == "shp-rlhf":
        from federatedscope.llm.dataloader.shp import \
            load_rlhf_dataset, SHP_PROMPT_DICT

        data_root = os.path.join(config.data.root, 'shp')
        list_train_prompts, _, _ = load_rlhf_dataset(data_root,
                                                     tokenizer=None,
                                                     max_num_test=1000)
        generation_prompt = SHP_PROMPT_DICT["shp"]
        selector_prompt = SHP_PROMPT_DICT["shp_cmp"]

    elif dataset_name.lower() == "shp-safe":
        from federatedscope.llm.dataloader.shp import \
            load_safe_dataset, SHP_PROMPT_DICT

        data_root = os.path.join(config.data.root, 'shp')
        list_train_prompts, _, _ = load_safe_dataset()
        generation_prompt = SHP_PROMPT_DICT["shp"]
        selector_prompt = SHP_PROMPT_DICT["shp_cmp"]

    return (data_root, list_train_prompts, generation_prompt, selector_prompt)


def get_input_data(list_data_dict, w=5):
    """
    Yield batches of data for generation.
    Reduced default batch size to 5 to avoid OOM during text generation.
    """
    total_batches = (len(list_data_dict) + w - 1) // w
    logger.info(f"Total batches to generate: {total_batches}, batch size: {w}, total prompts: {len(list_data_dict)}")
    for left in tqdm(range(0, len(list_data_dict), w), desc="Generating text pairs", total=total_batches, miniters=1, mininterval=10.0):
        yield list_data_dict[left:left + w]


class RLHF_finetuning:
    """
    Implementation of RLHF server
    """
    def __init__(
        self,
        model,
        tokenizer,
        config=None,
        selector_model=None,
        selector_tokenizer=None,
        generator_tokenizer=None,
        device="cpu",
        **kwargs,
    ):
        # obtain RLHF input data
        (
            self.data_root,
            self.list_train_prompts,
            self.generation_prompt,
            self.selector_prompt,
        ) = get_rlhf_prompts_dataset(config)

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.selector_model = selector_model
        self.selector_tokenizer = selector_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.device = device
        self._monitor = Monitor(config, monitored_object=self)

    def load_pairwise_data(self, saveto=None):
        # Name of a file saving the generated texts of original model
        _, model_name = self.config.model.type.split("@")[0].split('/', 1)
        dataset_name, _ = self.config.data.type.split("@")
        num_comp = max(2, self.config.llm.num_completions)
        
        # Extract tid from saveto if provided, otherwise from config.federate.save_to
        tid_suffix = ""
        if saveto:
            # Extract tid from saveto (e.g., "hhrl_rlhf_gemma_choice_gemma_fedbiscuit_u3_20000.ckpt" -> "20000")
            import re
            tid_match = re.search(r'_(\d+)\.ckpt$', saveto)
            if tid_match:
                tid_suffix = f"_{tid_match.group(1)}"
        elif hasattr(self.config, 'federate') and hasattr(self.config.federate, 'save_to'):
            # Try to extract tid from save_to path
            import re
            tid_match = re.search(r'_(\d+)\.ckpt$', self.config.federate.save_to)
            if tid_match:
                tid_suffix = f"_{tid_match.group(1)}"
        
        gen_fp = os.path.join(
            self.data_root,
            f"rlhf_pair_data_{model_name}_{dataset_name}_{num_comp}{tid_suffix}.json")

        if os.path.exists(gen_fp):
            # load the file with generated responses
            logger.info(f"Found existing generated text pairs file: {gen_fp}")
            with open(gen_fp, "r", encoding='utf-8') as f:
                list_pairwise_data = json.load(f)
            logger.info(f"Successfully loaded {len(list_pairwise_data)} text pairs "
                        f"from {os.path.abspath(gen_fp)}")
            logger.info("Skipping text generation. To regenerate, delete the file above.")
        else:
            # generate the output
            logger.info(f"The generated text file does not exist. "
                        f"Create a new one. Total prompts: {len(self.list_train_prompts)}")
            logger.info("Starting text generation (this may take a while)...")
            
            # Limit the number of prompts for faster generation during testing
            # You can adjust this or remove it for full dataset
            max_prompts = getattr(self.config, 'max_prompts_for_generation', len(self.list_train_prompts))
            prompts_to_use = self.list_train_prompts[:max_prompts]
            if len(prompts_to_use) < len(self.list_train_prompts):
                logger.info(f"Limiting generation to {max_prompts} prompts for faster processing")
            
            list_pairwise_data = self._generate_pairwise_data(
                prompts_to_use,
                self.model,
                self.generator_tokenizer,
                self.generation_prompt,
                max_new_tokens=self.config.llm.max_new_token,
                num_completions=self.config.llm.num_completions)

            # Save the data to a file with indentation for readability
            logger.info(f"Saving generated text pairs to {gen_fp}...")
            with open(gen_fp, "w", encoding='utf-8') as f:
                json.dump(list_pairwise_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"The generation process is done, and saved "
                        f"to {gen_fp}. Generated {len(list_pairwise_data)} pairs")
            logger.info(f"You can view the generated pairs in: {os.path.abspath(gen_fp)}")

        return list_pairwise_data

    def load_selector_preference_data(self, saveto, early_exiting=False):
        # This file save selector's choices
        fp = os.path.join(self.data_root, f"generated_choose_{saveto}.json")

        if os.path.exists(fp):
            logger.info(f"Found existing selector preference data file: {fp}")
            with open(fp, "r", encoding='utf-8') as f:
                list_preference_data = json.load(f)
            logger.info(f"Successfully loaded {len(list_preference_data)} preference data "
                        f"from {os.path.abspath(fp)}")
            logger.info("Skipping preference selection. To regenerate, delete the file above.")
        else:
            list_pairwise_data = self.load_pairwise_data(saveto=saveto)

            # choose the better one based on the given output
            logger.info("Select the better response.")
            list_preference_data = self._choose_better_response(
                list_pairwise_data,
                self.selector_model,
                self.selector_tokenizer,
                self.selector_prompt,
            )
            logger.info(list_preference_data[0])
            # save the choice to a file with indentation for readability
            with open(fp, "w", encoding='utf-8') as f:
                json.dump(list_preference_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved the selection results to {os.path.abspath(fp)}")

            if early_exiting:
                # For choosing the answer
                exit(0)

        return list_preference_data

    def train(self, saveto=None, early_exiting=False):
        if saveto is None:
            _, saveto = os.path.split(self.config.federate.save_to)
        # The training data should be selector's preference data
        list_train_dict = self.load_selector_preference_data(
            saveto, early_exiting)

        # move selector model to cpu
        # Check if model has meta tensors and handle them properly
        try:
            # First, ensure all parameters are on actual device (not meta)
            has_meta = False
            for param in self.selector_model.parameters():
                if param.is_meta:
                    has_meta = True
                    break
            
            if not has_meta:
                self.selector_model.cpu()
            else:
                logger.warning("Selector model contains meta tensors, skipping .cpu() call")
        except Exception as e:
            logger.warning(f"Error moving selector model to CPU: {e}, continuing...")
        
        gc.collect()
        torch.cuda.empty_cache()

        # load comparison dataset
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            self.tokenizer,
            prompt_input=self.generation_prompt,
            prompt_no_input=self.generation_prompt,
            output_A="output_A",
            output_B="output_B",
            choice="choice",
        )
        
        # Create ClientData and ensure it's properly initialized
        data = ClientData(self.config, train_dataset, None, None)
        # Ensure setup() is called after super().__init__() to create 'train' key in the dict
        if 'train' not in data:
            data.setup(self.config)
        
        # Verify train key exists
        if 'train' not in data:
            logger.error(f"Failed to create 'train' key in ClientData. train_dataset length: {len(train_dataset) if train_dataset else 'None'}")
            raise ValueError("ClientData does not have 'train' key after setup. Check train_dataset.")

        # create DPO trainer
        self.trainer = DPORewardTrainer(
            self.model,
            data,
            self.device,
            self.config,
            only_for_eval=False,
            monitor=self._monitor,
        )

        # start training
        for r in range(self.config.federate.total_round_num):
            logger.info("----------- Starting a new RLHF training round "
                        f"(Round #{r}) -------------")
            sample_size, model_para_all, results = self.trainer.train()
            train_log_res = self._monitor.format_eval_res(results,
                                                          rnd=r,
                                                          role="Server",
                                                          return_raw=True)
            logger.info(train_log_res)
            
            # Log to wandb if enabled
            if self.config.wandb.use and self.config.wandb.online_track:
                try:
                    import wandb
                    wandb_metrics = {}
                    
                    # Log loss metrics
                    if 'train_loss' in results:
                        wandb_metrics['train/loss'] = results['train_loss']
                    if 'train_avg_loss' in results:
                        wandb_metrics['train/avg_loss'] = results['train_avg_loss']
                    
                    # Log accuracy metrics separately for better visualization
                    if 'train_acc' in results:
                        wandb_metrics['train/accuracy'] = results['train_acc']
                    
                    # Use round as step for x-axis in wandb
                    if wandb_metrics:
                        wandb.log(wandb_metrics, step=r)
                except ImportError:
                    logger.warning("wandb not installed, skipping metrics logging")
                except Exception as e:
                    logger.warning(f"Failed to log metrics to wandb: {e}")
            # Save the checkpoint
            if (r + 1) % self.config.federate.save_freq == 0:
                if saveto in self.config.federate.save_to:
                    path = add_prefix_to_path(f"{r + 1}_",
                                              self.config.federate.save_to)
                else:
                    path = add_prefix_to_path(f"{r + 1}_{saveto}_",
                                              self.config.federate.save_to)
                self.model.save_model(path=path, state=r)

    def _generate_pairwise_data(self,
                                list_data_dict,
                                model,
                                tokenizer,
                                prompt,
                                max_new_tokens=60,
                                num_completions=2):
        generate_kwargs = dict(
            top_p=1.0,
            temperature=0.7,
            do_sample=True,
            # early_stopping=True,
            max_new_tokens=max_new_tokens,
            num_return_sequences=max(2, num_completions),
        )

        # Use self.device instead of detecting from model (which may be on CPU)
        model_device = self.device if hasattr(self, 'device') else _get_model_device(model)
        
        # Ensure model is on the correct device
        if isinstance(model_device, str):
            model_device = torch.device(model_device)
        
        logger.info(f"Model device: {model_device}, Generating text pairs for {len(list_data_dict)} prompts")
        
        # Move model to device if not already there
        try:
            current_device = next(model.parameters()).device
            if current_device != model_device:
                logger.info(f"Moving model from {current_device} to {model_device}")
                model = model.to(model_device)
        except Exception as e:
            logger.warning(f"Could not move model to device {model_device}: {e}, using current device")
            model_device = _get_model_device(model)

        new_list_data_dict = []
        batch_idx = 0
        total_batches = (len(list_data_dict) + 4) // 5
        for input_data in get_input_data(list_data_dict, w=5):  # Reduce batch size from 10 to 5 to avoid OOM
            batch_idx += 1
            # Log batch info only for first batch or every 20 batches
            log_batch_details = (batch_idx == 1 or batch_idx % 20 == 0)
            
            if log_batch_details:
                logger.info(f"Processing batch {batch_idx}/{total_batches}, batch size: {len(input_data)}")
            
            try:
                input_texts = [prompt.format_map(data) for data in input_data]
                input_text_tokens = tokenizer(
                    input_texts,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(model_device)

                if log_batch_details:
                    logger.info(f"Starting generation for batch {batch_idx}, input_ids shape: {input_text_tokens['input_ids'].shape}, "
                               f"max_new_tokens: {max_new_tokens}, num_return_sequences: {generate_kwargs['num_return_sequences']}")
                
                # Clear cache before generation
                if torch.cuda.is_available() and 'cuda' in str(model_device):
                    torch.cuda.empty_cache()
                    if log_batch_details:
                        logger.info(f"GPU memory before generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                # Ensure model is in eval mode
                model.eval()
                
                if log_batch_details:
                    logger.info(f"Calling model.generate() for batch {batch_idx} on device {model_device}...")
                output_ids = model.generate(**input_text_tokens, **generate_kwargs)
                
                if log_batch_details:
                    logger.info(f"Completed generation for batch {batch_idx}, output_ids shape: {output_ids.shape}")
                    if torch.cuda.is_available() and 'cuda' in str(model_device):
                        logger.info(f"GPU memory after generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                responses = tokenizer.batch_decode(output_ids,
                                                   skip_special_tokens=True,
                                                   ignore_tokenization_space=True)

                response_map = [[] for _ in input_data]
                for res in responses:
                    for idx, input_text in enumerate(input_texts):
                        if input_text in res:
                            gen_res = res.replace(input_text, "").strip()
                            response_map[idx].append(gen_res.replace("</s>", ""))
                            # response_map[idx].append(
                            #     " " + gen_res.replace("</s>", ""))
                            break

                # Log only first sample of first batch for debugging, rest just summary
                log_sample_details = (batch_idx == 1 and len(new_list_data_dict) == 0)
                
                for i, data in enumerate(input_data):
                    if log_sample_details and i == 0:
                        # Log first sample of first batch in detail (truncated)
                        data_preview = str(data)[:300] + "..." if len(str(data)) > 300 else str(data)
                        logger.info(f"Sample data (first of batch {batch_idx}, preview): {data_preview}")
                        for j, res in enumerate(response_map[i]):
                            # Truncate long responses in log
                            res_preview = res[:200] + "..." if len(res) > 200 else res
                            logger.info(f'Generated {j}-th response (preview): {res_preview}')
                    elif batch_idx % 20 == 0 and i == 0:
                        # Log summary every 20 batches
                        logger.info(f"Batch {batch_idx}: Processed {len(input_data)} samples, "
                                   f"generated {len(response_map[i])} responses per sample")

                    for output_A, output_B in combinations(response_map[i], 2):
                        new_data = copy.deepcopy(data)
                        new_data["output_A"] = output_A
                        new_data["output_B"] = output_B
                        new_list_data_dict.append(new_data)
                        
                # Clear cache after generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx} generation: {e}", exc_info=True)
                # Skip this batch and continue
                continue

        return new_list_data_dict

    @torch.no_grad()
    def _choose_better_response(self, list_data_dict, model, tokenizer,
                                prompt):
        choices = [tokenizer(f": {c}")["input_ids"][-1] for c in ["A", "B"]]
        logger.info(f'Choice indices: {choices}')

        for sample in list_data_dict:
            sample["fake_choice"] = random.choice([" A", " B"])

        token_dataset = LLMDataset(
            list_data_dict,
            tokenizer,
            prompt_input=prompt,
            prompt_no_input=prompt,
            output_tag="fake_choice",
        )
        dataloader = DataLoader(
            dataset=token_dataset,
            batch_size=10,
            shuffle=False,
            collate_fn=LLMDataCollator(tokenizer=tokenizer),
        )

        model_device = _get_model_device(model)

        predicted_indices = []
        if hasattr(model, "adapter_names") is False or len(
                model.adapter_names) == 1:
            # No adapter or only one LoRA adapter
            for idx, data_batch in enumerate(tqdm(dataloader)):
                input_ids = data_batch["input_ids"].to(model_device)
                labels = data_batch["labels"].to(model_device)
                attention_mask = data_batch["attention_mask"].to(model_device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                _, _, predicted, _ = cal_acc(outputs.logits, labels, choices)
                predicted_indices += predicted.tolist()
        else:
            # More than one adapters (exclude "default" one)
            for idx, data_batch in enumerate(tqdm(dataloader)):
                input_ids = data_batch["input_ids"].to(model_device)
                labels = data_batch["labels"].to(model_device)
                attention_mask = data_batch["attention_mask"].to(model_device)
                collective_choices = []
                for name in model.adapter_names:
                    if name == "default":
                        continue
                    model.set_active_adapter(name)
                    model.eval()
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                    _, _, predicted, _ = cal_acc(outputs.logits, labels,
                                                 choices)
                    collective_choices.append(predicted.tolist())
                array = np.array(collective_choices).T
                predicted_indices += [
                    np.bincount(array[i]).argmax().item()
                    for i in range(len(array))
                ]

        for choice, sample in zip(predicted_indices, list_data_dict):
            sample["choice"] = choice
            sample.pop("fake_choice", None)

        return list_data_dict

    def dpo_better_response(self):
        # generate the output
        list_train_dict = self._generate_pairwise_data(
            self.list_train_prompts,
            self.model,
            self.generator_tokenizer,
            self.generation_prompt,
            max_new_tokens=self.config.llm.max_new_token,
        )

        return self._dpo_better_response(list_train_dict, self.model,
                                         self.tokenizer,
                                         self.generation_prompt)

    @torch.no_grad()
    def _dpo_better_response(self, list_data_dict, model, tokenizer, prompt):
        for sample in list_data_dict:
            sample["fake_choice"] = random.choice([0, 1])

        dataset = LLMComparisonDataset(
            list_data_dict,
            tokenizer,
            prompt_input=prompt,
            prompt_no_input=prompt,
            output_A="output_A",
            output_B="output_B",
            choice="fake_choice",
        )


        model_device = _get_model_device(model)
        dataloader = DataLoader(dataset)

        predicted_indices = []
        for idx, data_batch in enumerate(tqdm(dataloader)):
            win_input_ids = data_batch["win_input_ids"].to(model_device)
            win_labels = data_batch["win_labels"].to(model_device)
            win_attention_mask = data_batch["win_attention_mask"].to(model_device)
            ref_win_outputs = model(
                disable_adapter=True,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask,
            )
            ref_win_logps = _get_batch_logps(ref_win_outputs.logits,
                                             win_labels,
                                             average_log_prob=False)
            policy_win_outputs = model(
                disable_adapter=False,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask,
            )
            policy_win_logps = _get_batch_logps(policy_win_outputs.logits,
                                                win_labels,
                                                average_log_prob=False)

            lose_input_ids = data_batch["lose_input_ids"].to(model_device)
            lose_labels = data_batch["lose_labels"].to(model_device)
            lose_attention_mask = data_batch["lose_attention_mask"].to(
                model_device)
            ref_lose_outputs = model(
                disable_adapter=True,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask,
            )
            ref_lose_logps = _get_batch_logps(ref_lose_outputs.logits,
                                              lose_labels,
                                              average_log_prob=False)
            policy_lose_outputs = model(
                disable_adapter=False,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask,
            )
            policy_lose_logps = _get_batch_logps(policy_lose_outputs.logits,
                                                 lose_labels,
                                                 average_log_prob=False)

            # DPO for reward calculation
            _, win_rewards, lose_rewards = dpo_loss(
                policy_win_logps,
                policy_lose_logps,
                ref_win_logps,
                ref_lose_logps,
                beta=1.0,
            )

            predicted = torch.where(
                win_rewards.cpu() > lose_rewards.cpu(),
                torch.zeros(len(win_input_ids)),
                torch.ones(len(win_input_ids)),
            )
            predicted_indices += predicted.tolist()

        for choice, sample in zip(predicted_indices, list_data_dict):
            sample["choice"] = choice
            sample.pop("fake_choice", None)

        return list_data_dict
