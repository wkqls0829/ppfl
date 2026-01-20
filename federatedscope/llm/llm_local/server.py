import logging
import torch
import random
import math
import numpy as np
from collections import defaultdict
from federatedscope.core.message import Message

from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.utils import merge_param_dict

logger = logging.getLogger(__name__)


class LLMMultiLoRAServer(Server):
    """
    Server implementation
    We broadcast the model to each client and ask them to train locally
    Afterward, we collect the model back and save it as checkpoints
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(LLMMultiLoRAServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        if self._cfg.llm.adapter.count > 1:
            self.aggregator.total_train_size = len(data.train_data)

            self.aggregator.num_clients = client_num

        if self._cfg.llm.adapter.local_only:
            logger.warning("In local training mode, we will use all clients. "
                           "And we set the total round to 0 for one training "
                           "round only. ")

            self.sampler = None
            self.sample_client_num = client_num

        if self._cfg.llm.adapter.grouping.use:
            self.msg_buffer['adapter_eval'] = dict()
        
        # VPL-GP related attributes
        self.vpl_gp_prior_mus = None
        self.vpl_gp_prior_logvars = None
        self.vpl_gp_prior_weights = None
        self.vpl_orthogonal_client_labels = None
        self.client_z_values_dict = defaultdict(list)  # {client_id: [z_values]}
        self.client_orthogonal_prototypes_dict = {}  # {client_id: prototypes}

    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers('grouping', self.callback_funcs_for_grouping,
                               ['set_active_adapter_idx'])

    def _start_new_training_round(self, aggregated_num=0, skip_grouping=False):
        if self._cfg.llm.adapter.grouping.use and not skip_grouping:
            total_warmup_round = 0
            if self._cfg.llm.adapter.warmup.use:
                warmup_round = self._cfg.llm.adapter.warmup.round
                total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count

            r = self._cfg.llm.adapter.grouping.round
            if self.state >= total_warmup_round and \
                    (self.state - total_warmup_round) % r == 0:
                logger.info('Server: Performing a grouping step...')
                self.broadcast_model_para(msg_type='adapter_eval',
                                          filter_unseen_clients=False)
                return

        super()._start_new_training_round(aggregated_num)

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            # merged_adapter = dict()

            # for client_id in train_msg_buffer.keys():
            #     if self.model_num == 1:
            #         _, model_param = train_msg_buffer[client_id]
            #         # merged_adapter.update(model_param)
            #         for key, value in model_param.items():
            #             if key not in merged_adapter:
            #                 merged_adapter[key] = [value]
            #             else:
            #                 merged_adapter[key].append(value)
            #     else:
            #         train_data_size, model_para_multiple = \
            #             train_msg_buffer[client_id]
            #         # merged_adapter.update(model_para_multiple[model_idx])
            #         for key, value in model_para_multiple[model_idx].items():
            #             if key not in merged_adapter:
            #                 merged_adapter[key] = [value]
            #             else:
            #                 merged_adapter[key].append(value)

            # # calculate the mean
            # for key in merged_adapter.keys():
            #     # logger.info(f'{key}: {len(merged_adapter[key])}')
            #     avg_tensor = torch.zeros_like(merged_adapter[key][0])
            #     for value in merged_adapter[key]:
            #         avg_tensor += (value / len(merged_adapter[key]))
            #     merged_adapter[key] = avg_tensor

            # msg_list = [(1, merged_adapter)]

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
            }
            # logger.info(f'The staleness is {staleness}')

            warmup_round = self._cfg.llm.adapter.warmup.round
            total_warmup_round = \
                warmup_round * self._cfg.llm.adapter.count
            if self._cfg.llm.adapter.warmup.use and \
                    self.state < total_warmup_round:
                result = aggregator.aggregate(agg_info)
            else:
                result = aggregator.aggregate_on_model(agg_info)

            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)
        
        # VPL-GP: Collect z distributions from clients (only if GP prior is enabled)
        if hasattr(self._cfg.llm, 'vpl_use_gp_prior') and self._cfg.llm.vpl_use_gp_prior:
            self._collect_vpl_gp_prior_distributions()
            self._compute_balanced_orthogonal_labels()
        
        # Always collect z values for visualization (even if GP prior is disabled)
        # This allows t-SNE visualization for all VPL experiments
        if hasattr(self._cfg.llm, 'vpl_latent_dim'):  # VPL is enabled
            self._collect_z_values_for_visualization()
        
        return aggregated_num
    
    def merge_eval_results_from_all_clients(self):
        """
        Override to add VPL-specific wandb logging for aggregated results.
        """
        # Call parent method to get aggregated results
        formatted_logs_all_set = super().merge_eval_results_from_all_clients()
        
        # Log VPL metrics to wandb if enabled (server-side aggregated logging)
        if (self._cfg.wandb.use and self._cfg.wandb.online_track and 
            hasattr(self._cfg.llm, 'vpl_latent_dim')):  # VPL is enabled
            try:
                import wandb
                
                # Extract VPL metrics from aggregated results
                round = max(self.msg_buffer['eval'].keys())
                eval_msg_buffer = self.msg_buffer['eval'][round]
                
                # Collect VPL metrics from all clients
                vpl_metrics_all_clients = {
                    'vpl_total_loss': [],
                    'vpl_reconstruction_loss': [],
                    'vpl_kl_loss': [],
                    'vpl_orthogonal_loss': []
                }
                client_ids = []
                
                for client_id in eval_msg_buffer:
                    if eval_msg_buffer[client_id] is None:
                        continue
                    if client_id in self.unseen_clients_id:
                        continue  # Skip unseen clients for aggregated metrics
                    
                    client_results = eval_msg_buffer[client_id]
                    client_ids.append(client_id)
                    
                    if 'loss' in client_results:
                        vpl_metrics_all_clients['vpl_total_loss'].append(float(client_results['loss']))
                    if 'vpl_reconstruction_loss' in client_results:
                        vpl_metrics_all_clients['vpl_reconstruction_loss'].append(float(client_results['vpl_reconstruction_loss']))
                    if 'vpl_kl_loss' in client_results:
                        vpl_metrics_all_clients['vpl_kl_loss'].append(float(client_results['vpl_kl_loss']))
                    if 'vpl_orthogonal_loss' in client_results:
                        vpl_metrics_all_clients['vpl_orthogonal_loss'].append(float(client_results['vpl_orthogonal_loss']))
                
                # Check if this is HRL dataset for reward model metrics
                dataset_type = getattr(self._cfg.data, 'type', '').lower()
                is_hrl = 'hh-rlhf' in dataset_type or 'hrl' in dataset_type
                
                # Collect reward model metrics for HRL
                reward_metrics_all_clients = {
                    'avg_harmlessness': [],
                    'avg_helpfulness': [],
                    'helpfulness_winrate': [],
                    'harmlessness_winrate': []
                }
                if is_hrl:
                    for client_id in eval_msg_buffer:
                        if eval_msg_buffer[client_id] is None:
                            continue
                        if client_id in self.unseen_clients_id:
                            continue
                        
                        client_results = eval_msg_buffer[client_id]
                        if 'avg_harmlessness' in client_results:
                            reward_metrics_all_clients['avg_harmlessness'].append(float(client_results['avg_harmlessness']))
                        if 'avg_helpfulness' in client_results:
                            reward_metrics_all_clients['avg_helpfulness'].append(float(client_results['avg_helpfulness']))
                        if 'helpfulness_winrate' in client_results:
                            reward_metrics_all_clients['helpfulness_winrate'].append(float(client_results['helpfulness_winrate']))
                        if 'harmlessness_winrate' in client_results:
                            reward_metrics_all_clients['harmlessness_winrate'].append(float(client_results['harmlessness_winrate']))
                
                # Log aggregated metrics (averaged over all clients)
                wandb_metrics = {}
                if vpl_metrics_all_clients['vpl_total_loss']:
                    wandb_metrics['server/train/vpl_total_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_total_loss'])
                if vpl_metrics_all_clients['vpl_reconstruction_loss']:
                    wandb_metrics['server/train/vpl_reconstruction_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_reconstruction_loss'])
                if vpl_metrics_all_clients['vpl_kl_loss']:
                    wandb_metrics['server/train/vpl_kl_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_kl_loss'])
                if vpl_metrics_all_clients['vpl_orthogonal_loss']:
                    wandb_metrics['server/train/vpl_orthogonal_loss_avg'] = np.mean(vpl_metrics_all_clients['vpl_orthogonal_loss'])
                
                # Log reward model metrics for HRL (averaged)
                if is_hrl:
                    if reward_metrics_all_clients['avg_harmlessness']:
                        wandb_metrics['server/train/avg_harmlessness_avg'] = np.mean(reward_metrics_all_clients['avg_harmlessness'])
                    if reward_metrics_all_clients['avg_helpfulness']:
                        wandb_metrics['server/train/avg_helpfulness_avg'] = np.mean(reward_metrics_all_clients['avg_helpfulness'])
                    if reward_metrics_all_clients['helpfulness_winrate']:
                        wandb_metrics['server/train/helpfulness_winrate_avg'] = np.mean(reward_metrics_all_clients['helpfulness_winrate'])
                    if reward_metrics_all_clients['harmlessness_winrate']:
                        wandb_metrics['server/train/harmlessness_winrate_avg'] = np.mean(reward_metrics_all_clients['harmlessness_winrate'])
                
                # Log individual client metrics (for designated clients)
                # Log first 3 clients as designated clients (or all if less than 3)
                designated_clients = client_ids[:min(3, len(client_ids))]
                for client_id in designated_clients:
                    if eval_msg_buffer[client_id] is None:
                        continue
                    client_results = eval_msg_buffer[client_id]
                    
                    if 'loss' in client_results:
                        wandb_metrics[f'client_{client_id}/train/vpl_total_loss'] = float(client_results['loss'])
                    if 'vpl_reconstruction_loss' in client_results:
                        wandb_metrics[f'client_{client_id}/train/vpl_reconstruction_loss'] = float(client_results['vpl_reconstruction_loss'])
                    if 'vpl_kl_loss' in client_results:
                        wandb_metrics[f'client_{client_id}/train/vpl_kl_loss'] = float(client_results['vpl_kl_loss'])
                    if 'vpl_orthogonal_loss' in client_results:
                        wandb_metrics[f'client_{client_id}/train/vpl_orthogonal_loss'] = float(client_results['vpl_orthogonal_loss'])
                    
                    # Log reward model metrics for HRL (individual clients)
                    if is_hrl:
                        if 'avg_harmlessness' in client_results:
                            wandb_metrics[f'client_{client_id}/train/avg_harmlessness'] = float(client_results['avg_harmlessness'])
                        if 'avg_helpfulness' in client_results:
                            wandb_metrics[f'client_{client_id}/train/avg_helpfulness'] = float(client_results['avg_helpfulness'])
                        if 'helpfulness_winrate' in client_results:
                            wandb_metrics[f'client_{client_id}/train/helpfulness_winrate'] = float(client_results['helpfulness_winrate'])
                        if 'harmlessness_winrate' in client_results:
                            wandb_metrics[f'client_{client_id}/train/harmlessness_winrate'] = float(client_results['harmlessness_winrate'])
                
                if wandb_metrics:
                    wandb.log(wandb_metrics, step=round)
                    logger.info(f"Logged VPL metrics to wandb for round {round}: {len(wandb_metrics)} metrics")
                    
            except ImportError:
                logger.warning("wandb not installed, skipping VPL metrics logging")
            except Exception as e:
                logger.warning(f"Failed to log VPL metrics to wandb: {e}")

        return formatted_logs_all_set

    def trigger_for_start(self):
        # start feature engineering (This part is for hard code)
        if self.check_client_join_in():
            logger.info('Waited all clients join, start now...')
            # Only send adapter_eval message if grouping is enabled
            if self._cfg.llm.adapter.grouping.use:
                self.trigger_for_feat_engr(self.broadcast_model_para, {
                    'msg_type': 'adapter_eval',
                    'filter_unseen_clients': False,
            })
                logger.info('Server: Performing a grouping step...')
            else:
                # If grouping is not enabled, start training round directly
                logger.info(
                    '----------- Starting training (Round #{:d}) -------------'.
                    format(self.state))
                self._start_new_training_round()

    def callback_funcs_for_grouping(self, message: Message):
        rnd = message.state
        sender = message.sender
        content = message.content

        # Only process adapter_eval if grouping is enabled
        if not self._cfg.llm.adapter.grouping.use:
            return False
        
        # Initialize adapter_eval if not already initialized
        if 'adapter_eval' not in self.msg_buffer:
            self.msg_buffer['adapter_eval'] = dict()
        
        if rnd not in self.msg_buffer['adapter_eval'].keys():
            self.msg_buffer['adapter_eval'][rnd] = dict()

        self.msg_buffer['adapter_eval'][rnd][sender] = \
            [(i, content[f'adapter_{i}_avg_loss'])
             for i in range(self._cfg.llm.adapter.count)]
        self.msg_buffer['adapter_eval'][rnd][sender] = \
            sorted(self.msg_buffer['adapter_eval'][rnd][sender],
                   key=lambda x: x[1])

        return self.check_and_grouping()

    def check_and_grouping(self):
        if 'adapter_eval' not in self.msg_buffer.keys() or \
                len(self.msg_buffer['adapter_eval'].keys()) == 0:
            return False

        buffer = self.msg_buffer['adapter_eval']
        cur_round = max(buffer.keys())
        cur_buffer = buffer[cur_round]
        if len(cur_buffer) < self.client_num:
            return False

        # convert the list to the iterator
        for sender in cur_buffer.keys():
            cur_buffer[sender] = iter(cur_buffer[sender])

        num_adap = self._cfg.llm.adapter.count
        self.adapter_grouping = dict()
        adapter_grouping = {i: [] for i in range(num_adap)}
        senders = [sender for sender in cur_buffer.keys()]
        random.shuffle(senders)
        unassigned_client_num = len(senders)
        while unassigned_client_num > 0:
            num_finished = len(self.adapter_grouping)
            max_size = math.ceil(unassigned_client_num /
                                 (num_adap - num_finished))

            # step 1: Assign to the adapter where the clients
            # well performs
            for sender in senders:
                adap_idx, loss = next(cur_buffer[sender])
                while adap_idx not in adapter_grouping:
                    adap_idx, loss = next(cur_buffer[sender])
                adapter_grouping[adap_idx].append(sender)

            # step 2: Find the adapter with the most clients
            max_adap_idx_size = [0, 0]
            for adap_idx, candidates in adapter_grouping.items():
                if len(candidates) > max_adap_idx_size[1]:
                    max_adap_idx_size = [adap_idx, len(candidates)]

            # step 3: If the number of candidates is greater than
            # max_size, preserve the first max_size
            adap_idx = max_adap_idx_size[0]
            candidates = adapter_grouping[adap_idx][:max_size]

            # step 4: update the senders list, remove the selected
            # adapter from adapter_grouping
            senders = adapter_grouping[adap_idx][max_size:]
            self.adapter_grouping[adap_idx] = candidates
            adapter_grouping.pop(adap_idx)
            unassigned_client_num -= len(self.adapter_grouping[adap_idx])
            logger.info(f'Adapter {adap_idx} is done with the clients '
                        f'{self.adapter_grouping[adap_idx]}')

        # broadcast the new grouping info to all clients
        for adap_idx, receiver in self.adapter_grouping.items():
            self.comm_manager.send(
                Message(msg_type='set_active_adapter_idx',
                        sender=self.ID,
                        receiver=receiver,
                        state=self.state,
                        timestamp=self.cur_timestamp,
                        content=adap_idx))

        # resume the training based on the new group...
        self._start_new_training_round(skip_grouping=True)

        return True  # move_on_flag
    
    def _compute_manual_orthogonal_labels(self, train_msg_buffer):
        """
        Assign manual orthogonal labels based on client data type.
        Harmless clients (first half) get label 0, helpful clients (second half) get label 1.
        Only assigns labels to clients that participated in this round.
        """
        if not (hasattr(self._cfg.llm, 'vpl_use_manual_orthogonal_labels') and 
                self._cfg.llm.vpl_use_manual_orthogonal_labels):
            return None
        
        # Get participating client IDs
        participating_clients = sorted(train_msg_buffer.keys())
        num_participants = len(participating_clients)
        
        if num_participants == 0:
            return {}
        
        # Assign labels: first half = 0 (harmless), second half = 1 (helpful)
        labels = {}
        split_point = num_participants // 2
        for idx, client_id in enumerate(participating_clients):
            if idx < split_point:
                labels[client_id] = 0
            else:
                labels[client_id] = 1
        
        logger.info(f"Assigned manual orthogonal labels: {labels}")
        return labels
    
    def _collect_vpl_gp_prior_distributions(self):
        """
        Collect z distributions (mu, logvar) from clients for VPL-GP mixture prior.
        Called after _perform_federated_aggregation.
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        
        client_mus = []
        client_logvars = []
        client_weights = []
        client_ids = []
        
        for client_id in train_msg_buffer.keys():
            if self.model_num == 1:
                _, model_para = train_msg_buffer[client_id]
            else:
                _, model_para_multiple = train_msg_buffer[client_id]
                model_para = model_para_multiple[0]  # Use first model
            
            # Extract z distribution from model parameters
            if 'client_z_mu' in model_para and 'client_z_logvar' in model_para:
                mu = model_para['client_z_mu']
                logvar = model_para['client_z_logvar']
                
                # Convert to tensors if needed
                if not isinstance(mu, torch.Tensor):
                    mu = torch.tensor(mu, dtype=torch.float32)
                if not isinstance(logvar, torch.Tensor):
                    logvar = torch.tensor(logvar, dtype=torch.float32)
                
                client_mus.append(mu)
                client_logvars.append(logvar)
                client_ids.append(client_id)
                
                # Use sample size as weight
                sample_size = model_para.get('sample_size', 1)
                client_weights.append(sample_size)
        
        if len(client_mus) == 0:
            return
        
        # Normalize weights
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        
        # Stack tensors
        client_mus = torch.stack(client_mus)  # (num_clients, latent_dim)
        client_logvars = torch.stack(client_logvars)  # (num_clients, latent_dim)
        client_weights = torch.tensor(client_weights, dtype=torch.float32)
        
        # Update prior (accumulate across rounds)
        if self.vpl_gp_prior_mus is None:
            # First round: initialize
            self.vpl_gp_prior_mus = client_mus
            self.vpl_gp_prior_logvars = client_logvars
            self.vpl_gp_prior_weights = client_weights
            self._vpl_gp_client_ids = client_ids  # Initialize client IDs
            updated_count = len(client_mus)
            from_previous = 0
        else:
            # Update existing clients and add new ones
            existing_client_ids = set(getattr(self, '_vpl_gp_client_ids', []))
            current_client_ids = set(client_ids)
            
            # Update existing
            updated_indices = []
            new_mus = []
            new_logvars = []
            new_weights = []
            new_ids = []
            
            for idx, cid in enumerate(client_ids):
                if cid in existing_client_ids:
                    # Update existing
                    old_idx = self._vpl_gp_client_ids.index(cid)
                    self.vpl_gp_prior_mus[old_idx] = client_mus[idx]
                    self.vpl_gp_prior_logvars[old_idx] = client_logvars[idx]
                    self.vpl_gp_prior_weights[old_idx] = client_weights[idx]
                    updated_indices.append(old_idx)
                else:
                    # New client
                    new_mus.append(client_mus[idx])
                    new_logvars.append(client_logvars[idx])
                    new_weights.append(client_weights[idx])
                    new_ids.append(cid)
            
            # Append new clients
            if len(new_mus) > 0:
                new_mus = torch.stack(new_mus)
                new_logvars = torch.stack(new_logvars)
                new_weights = torch.stack(new_weights)
                
                self.vpl_gp_prior_mus = torch.cat([self.vpl_gp_prior_mus, new_mus], dim=0)
                self.vpl_gp_prior_logvars = torch.cat([self.vpl_gp_prior_logvars, new_logvars], dim=0)
                self.vpl_gp_prior_weights = torch.cat([self.vpl_gp_prior_weights, new_weights], dim=0)
                
                # Update client ID list
                if not hasattr(self, '_vpl_gp_client_ids'):
                    self._vpl_gp_client_ids = []
                self._vpl_gp_client_ids.extend(new_ids)
            
            # Renormalize weights
            total_weight = self.vpl_gp_prior_weights.sum()
            if total_weight > 0:
                self.vpl_gp_prior_weights = self.vpl_gp_prior_weights / total_weight
            
            updated_count = len(updated_indices) + len(new_ids)
            from_previous = len(existing_client_ids) - len(updated_indices)
        
        # Store client IDs for tracking
        if not hasattr(self, '_vpl_gp_client_ids'):
            self._vpl_gp_client_ids = client_ids
        
        logger.info(f"Collected {len(client_mus)} client z distributions for VPL-GP prior. "
                   f"Total clients in prior: {len(self.vpl_gp_prior_mus)} "
                   f"(updated: {updated_count}, from previous rounds: {from_previous})")
    
    def _compute_balanced_orthogonal_labels(self):
        """
        Compute balanced orthogonal labels using k-means on z means.
        Uses manual labels if configured, otherwise uses k-means.
        For hh-rlhf dataset, k is fixed to 2.
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        
        # Check if manual labels are explicitly requested
        use_manual = (hasattr(self._cfg.llm, 'vpl_use_manual_orthogonal_labels') and 
                     self._cfg.llm.vpl_use_manual_orthogonal_labels)
        
        if use_manual:
            # Use manual labels if explicitly configured
            manual_labels = self._compute_manual_orthogonal_labels(train_msg_buffer)
            if manual_labels is not None:
                self.vpl_orthogonal_client_labels = manual_labels
                # Count labels for logging
                label_counts = defaultdict(int)
                for client_id, label in manual_labels.items():
                    label_counts[label] += 1
                logger.info(f"Computed manual orthogonal labels for {len(manual_labels)} clients "
                          f"at round {self.state}: {dict(label_counts)}")
                return
        
        # Otherwise, use k-means on z means
        if self.vpl_gp_prior_mus is None or len(self.vpl_gp_prior_mus) == 0:
            return
        
        try:
            from sklearn.cluster import KMeans
            
            # Get z means for participating clients
            participating_client_ids = sorted(train_msg_buffer.keys())
            if len(participating_client_ids) < 2:
                return
            
            # Map client IDs to indices in prior
            client_id_to_idx = {cid: idx for idx, cid in enumerate(self._vpl_gp_client_ids)}
            z_means = []
            valid_client_ids = []
            
            for client_id in participating_client_ids:
                if client_id in client_id_to_idx:
                    idx = client_id_to_idx[client_id]
                    z_means.append(self.vpl_gp_prior_mus[idx].cpu().numpy())
                    valid_client_ids.append(client_id)
            
            if len(z_means) < 2:
                return
            
            z_means = np.array(z_means)
            
            # Determine number of clusters (k)
            # For hh-rlhf dataset, k is fixed to 2
            dataset_type = getattr(self._cfg.data, 'type', '').lower()
            if 'hh-rlhf' in dataset_type or 'hrl' in dataset_type:
                n_clusters = 2
                logger.info(f"Using k=2 for hh-rlhf dataset")
            else:
                # Use config value, default to number of prototypes
                n_clusters = getattr(self._cfg.llm, 'vpl_num_prototypes', 2)
                logger.info(f"Using k={n_clusters} from config (vpl_num_prototypes)")
            
            # Ensure n_clusters doesn't exceed number of clients
            n_clusters = min(n_clusters, len(z_means))
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(z_means)
            
            # Create label dict
            self.vpl_orthogonal_client_labels = {
                client_id: int(label) for client_id, label in zip(valid_client_ids, labels)
            }
            
            # Count labels for logging
            label_counts = defaultdict(int)
            for label in labels:
                label_counts[int(label)] += 1
            
            logger.info(f"Computed k-means orthogonal labels (k={n_clusters}) for {len(valid_client_ids)} clients "
                      f"at round {self.state}: {dict(label_counts)}")
        except Exception as e:
            logger.warning(f"Failed to compute balanced orthogonal labels: {e}")
    
    def _collect_z_values_for_visualization(self):
        """
        Collect z values from clients for t-SNE visualization.
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        
        z_values_list = []
        client_ids_list = []
        
        for client_id in train_msg_buffer.keys():
            if self.model_num == 1:
                _, model_para = train_msg_buffer[client_id]
            else:
                _, model_para_multiple = train_msg_buffer[client_id]
                model_para = model_para_multiple[0]
            
            # Extract z values
            if 'client_z_values' in model_para:
                z_values = model_para['client_z_values']
                if isinstance(z_values, torch.Tensor):
                    z_values = z_values.cpu().numpy()
                elif isinstance(z_values, list):
                    z_values = np.array(z_values)
                
                if len(z_values.shape) == 1:
                    z_values = z_values.reshape(1, -1)
                
                z_values_list.append(z_values)
                client_ids_list.extend([client_id] * len(z_values))
        
        if len(z_values_list) == 0:
            return
        
        # Concatenate all z values
        all_z_values = np.concatenate(z_values_list, axis=0)
        
        # Store in dict
        for client_id in set(client_ids_list):
            client_z_mask = np.array(client_ids_list) == client_id
            client_z = all_z_values[client_z_mask]
            self.client_z_values_dict[client_id].extend(client_z.tolist())
        
        # Visualize every 10 rounds (or every round if configured)
        visualize_freq = getattr(self._cfg.llm, 'vpl_tsne_visualize_freq', 10)  # Default: every 10 rounds
        if self.state % visualize_freq == 0:
            self._visualize_cross_client_z()
        
        total_points = sum(len(v) for v in self.client_z_values_dict.values())
        unique_clients = len(self.client_z_values_dict)
        logger.info(f"Round {self.state}: Collected z values from {len(set(client_ids_list))} clients. "
                   f"Total accumulated: {total_points} points across {unique_clients} clients")
    
    def _visualize_cross_client_z(self):
        """
        Visualize cross-client z values using t-SNE.
        """
        try:
            from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z
            
            # Prepare z values and labels
            z_values_list = []
            client_labels_list = []
            orthogonal_labels_list = []
            
            for client_id, z_list in self.client_z_values_dict.items():
                if len(z_list) == 0:
                    continue
                
                client_z = np.array(z_list)
                z_values_list.append(client_z)
                client_labels_list.extend([client_id] * len(client_z))
                
                # Get orthogonal label if available
                if self.vpl_orthogonal_client_labels and client_id in self.vpl_orthogonal_client_labels:
                    orthogonal_labels_list.extend([self.vpl_orthogonal_client_labels[client_id]] * len(client_z))
                else:
                    orthogonal_labels_list.extend([-1] * len(client_z))
            
            if len(z_values_list) == 0:
                return
            
            all_z = np.concatenate(z_values_list, axis=0)
            
            # Get orthogonal prototypes if available
            orthogonal_prototypes = None
            if hasattr(self, 'client_orthogonal_prototypes_dict') and len(self.client_orthogonal_prototypes_dict) > 0:
                # Collect prototypes from all clients
                prototypes_list = []
                for client_id, prototypes in self.client_orthogonal_prototypes_dict.items():
                    if prototypes is not None:
                        if isinstance(prototypes, torch.Tensor):
                            prototypes = prototypes.cpu().numpy()
                        prototypes_list.append(prototypes)
                
                if len(prototypes_list) > 0:
                    orthogonal_prototypes = np.concatenate(prototypes_list, axis=0)
            
            visualize_cross_client_z(
                z_values=all_z,
                client_labels=client_labels_list,
                orthogonal_labels=orthogonal_labels_list if len(orthogonal_labels_list) > 0 else None,
                orthogonal_prototypes=orthogonal_prototypes,
                round_num=self.state,
                output_dir=self._cfg.outdir,
                wandb_project=self._cfg.wandb.name_project if self._cfg.wandb.use else None
            )
        except Exception as e:
            logger.warning(f"Failed to visualize cross-client z: {e}")
    
    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        Override to broadcast VPL-GP prior and orthogonal labels.
        """
        # Call parent method
        super().broadcast_model_para(
            msg_type=msg_type,
            sample_client_num=sample_client_num,
            filter_unseen_clients=filter_unseen_clients
        )
        
        # Broadcast VPL-GP prior if available
        if (hasattr(self._cfg.llm, 'vpl_use_gp_prior') and 
            self._cfg.llm.vpl_use_gp_prior and 
            self.vpl_gp_prior_mus is not None and
            self.state > 0):  # Don't broadcast at round 0
            
            # Prepare prior data
            prior_content = {
                'vpl_gp_prior_mus': self.vpl_gp_prior_mus.cpu().tolist(),
                'vpl_gp_prior_logvars': self.vpl_gp_prior_logvars.cpu().tolist(),
                'vpl_gp_prior_weights': self.vpl_gp_prior_weights.cpu().tolist(),
            }
            
            # Use same logic as parent class: sample if sample_client_num > 0, else broadcast to all
            if sample_client_num > 0:
                # Check if sampler is available and has idle clients
                if self.sampler is not None:
                    idle_clients = np.nonzero(self.sampler.client_state)[0]
                    if len(idle_clients) > 0:
                        selected_clients = self.sampler.sample(size=sample_client_num)
                    else:
                        # All clients are working, use all clients instead
                        selected_clients = list(self.comm_manager.neighbors.keys())
                        logger.warning(f"No idle clients available, broadcasting to all {len(selected_clients)} clients")
                else:
                    selected_clients = list(self.comm_manager.neighbors.keys())
            else:
                # Broadcast to all clients
                selected_clients = list(self.comm_manager.neighbors.keys())
            
            for receiver in selected_clients:
                self.comm_manager.send(
                    Message(msg_type='vpl_gp_prior',
                           sender=self.ID,
                           receiver=[receiver],
                           state=self.state,
                           timestamp=self.cur_timestamp,
                           content=prior_content))
            
            logger.info(f"Broadcasting VPL-GP prior with {len(self.vpl_gp_prior_mus)} client distributions to {len(selected_clients)} clients at round {self.state}")
        
        # Broadcast orthogonal labels if available
        if (self.vpl_orthogonal_client_labels is not None and
            self.state > 0):  # Don't broadcast at round 0
            
            # Use same logic as parent class: sample if sample_client_num > 0, else broadcast to all
            if sample_client_num > 0:
                # Check if sampler is available and has idle clients
                if self.sampler is not None:
                    idle_clients = np.nonzero(self.sampler.client_state)[0]
                    if len(idle_clients) > 0:
                        selected_clients = self.sampler.sample(size=sample_client_num)
                    else:
                        # All clients are working, use all clients instead
                        selected_clients = list(self.comm_manager.neighbors.keys())
                        logger.warning(f"No idle clients available, broadcasting to all {len(selected_clients)} clients")
                else:
                    selected_clients = list(self.comm_manager.neighbors.keys())
            else:
                # Broadcast to all clients
                selected_clients = list(self.comm_manager.neighbors.keys())
            
            for receiver in selected_clients:
        self.comm_manager.send(
                    Message(msg_type='vpl_orthogonal_labels',
                    sender=self.ID,
                           receiver=[receiver],
                           state=self.state,
                    timestamp=self.cur_timestamp,
                           content=self.vpl_orthogonal_client_labels))
            
            logger.info(f"Broadcasting orthogonal labels to {len(selected_clients)} clients at round {self.state}")