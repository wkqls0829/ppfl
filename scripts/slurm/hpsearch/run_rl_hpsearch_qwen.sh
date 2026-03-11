#!/bin/bash

# Hyperparameter Search RL Training Script (Qwen 2)
# TID range: 55100-55138 (RL), selector: 54100-54138

#SBATCH -p A6000,RTX6000ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 3-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

RL_TID=$1
SELECTOR_TID=$2

if [ -z "$RL_TID" ] || [ -z "$SELECTOR_TID" ]; then
    echo "Usage: $0 <rl_tid> <selector_tid>"
    echo "  rl_tid: RL Task ID (e.g., 55100)"
    echo "  selector_tid: Selector Task ID (e.g., 54100)"
    exit 1
fi

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
fi

if [ -d "/hdd/hdd3/kjb" ]; then
    DATA_ROOT="/hdd/hdd3/kjb"
    CHECKPOINT_BASE="/hdd/hdd3/kjb/checkpoints"
else
    DATA_ROOT="$WORK_DIR/data"
    CHECKPOINT_BASE="$WORK_DIR/checkpoints"
fi
mkdir -p "$DATA_ROOT" "$CHECKPOINT_BASE"
export HF_HOME="$WORK_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$WORK_DIR/.cache/huggingface/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

CHECKPOINT_DIR="$CHECKPOINT_BASE"
MODEL="qwen2"
METHOD="vplgp"

SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
if [ ! -f "$SELECTOR_CKPT" ]; then
    SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
fi
if [ ! -f "$SELECTOR_CKPT" ]; then
    SELECTOR_CKPT="$CHECKPOINT_DIR/40_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_ortho_t${SELECTOR_TID}.ckpt"
fi
if [ ! -f "$SELECTOR_CKPT" ]; then
    SELECTOR_CKPT="$CHECKPOINT_DIR/final_hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
fi
if [ ! -f "$SELECTOR_CKPT" ]; then
    SELECTOR_CKPT="$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_${METHOD}_t${SELECTOR_TID}.ckpt"
fi
if [ ! -f "$SELECTOR_CKPT" ]; then
    echo "ERROR: Qwen selector checkpoint not found for TID $SELECTOR_TID"
    ls -lh "$CHECKPOINT_DIR"/*${SELECTOR_TID}* 2>/dev/null | head -10 || true
    exit 1
fi
echo "✓ Using selector checkpoint: $SELECTOR_CKPT"

CONFIG_BASE="cfg/vpl-gp/hrl.yaml"
TRAINER="llmdporewardtrainer"
CONFIG_FILE="cfg/hpsearch/vpl-gp-rl-qwen/hrl_${RL_TID}.yaml"
mkdir -p $(dirname $CONFIG_FILE)

if [ ! -f "$CONFIG_BASE" ]; then
    echo "ERROR: Base config not found: $CONFIG_BASE"
    exit 1
fi
cp $CONFIG_BASE $CONFIG_FILE

python3 << EOF
import yaml
config_file = "$CONFIG_FILE"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
if 'llm' not in config:
    config['llm'] = {}
if 'cache' not in config['llm']:
    config['llm']['cache'] = {}
config['llm']['cache']['model'] = "$WORK_DIR/.cache/huggingface/transformers"
config['use_gpu'] = True
config['device'] = 0
if 'dataloader' not in config:
    config['dataloader'] = {}
config['dataloader']['num_workers'] = 0
config['federate']['client_num'] = 1
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_rlhf_${MODEL}_choice_${METHOD}_t${RL_TID}.ckpt"
config['data']['root'] = "$DATA_ROOT"
config['model']['type'] = 'Qwen/Qwen2-0.5B@huggingface_llm'
config['trainer']['type'] = "$TRAINER"
config['expname'] = "vplgp_hrl_ortho_qwen_t${RL_TID}"
config['train']['optimizer']['lr'] = 0.0001
config['llm']['grad_accum_step'] = 4
config['llm']['rlhf_use_variational_selection'] = True
config['llm']['rlhf_use_variational_generation'] = False
config['llm']['rlhf_selector_checkpoint'] = "$SELECTOR_CKPT"
config['llm']['vpl_latent_dim'] = 32
config['llm']['vpl_feature_method'] = 'choice_logits'
config['llm']['vpl_use_feature_difference'] = True
config['llm']['vpl_use_difference_only'] = True
config['llm']['vpl_use_gp_prior'] = True
config['llm']['vpl_gp_temperature'] = 1.0
config['llm']['reward_coeff'] = 0.1
config['llm']['max_prompts_for_generation'] = 50
config['llm']['generation_batch_size'] = 3
config['llm']['max_samples_for_reward'] = 30
config['llm']['use_gpt_api_for_winrate'] = True
config['llm']['use_baseline_model_for_winrate'] = True
config['llm']['openai_model'] = 'gpt-4o-mini'
if 'eval' not in config:
    config['eval'] = {}
config['eval']['freq'] = 10
config['eval']['use_gpt_api_for_winrate'] = True
config['eval']['use_baseline_model_for_winrate'] = True
config['eval']['metrics'] = ['loss', 'acc', 'helpfulness_winrate', 'harmlessness_winrate']
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
print(f"Config file created: {config_file}")
EOF

SELECTOR_CFG="cfg/hpsearch/vpl-gp-qwen/phase_${SELECTOR_TID}.yaml"
if [ ! -f "$SELECTOR_CFG" ]; then
    SELECTOR_CFG=""
fi

echo "Starting Qwen RL training: RL_TID=$RL_TID, Selector_TID=$SELECTOR_TID"
if [ -n "$SELECTOR_CFG" ]; then
    python -u federatedscope/llm/rlhf/main.py --cfg $CONFIG_FILE --selector-cfg-file $SELECTOR_CFG > outputs/${RL_TID}.log 2>&1
else
    python -u federatedscope/llm/rlhf/main.py --cfg $CONFIG_FILE > outputs/${RL_TID}.log 2>&1
fi
echo "RL experiment completed: TID=$RL_TID"
