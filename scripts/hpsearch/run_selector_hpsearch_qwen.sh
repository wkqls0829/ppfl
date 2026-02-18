#!/bin/bash

# Hyperparameter Search Selector Training Script (Qwen 2)
# SLURM cluster execution script
# TID range: 54100-54138 (selector experiments, Qwen2-0.5B)

#SBATCH -p A6000,RTX6000ADA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH -o /home2/jbkoo/slurm/logs/slurm-%A-%x.out
#SBATCH --exclude=n27,n33,n42,n72

# Parse arguments
TID=$1  # Task ID (e.g., 54100)

if [ -z "$TID" ]; then
    echo "Usage: $0 <tid>"
    echo "  tid: Task ID (e.g., 54100) for Qwen hpsearch"
    exit 1
fi

WORK_DIR="/home2/jbkoo/ppfl"
cd $WORK_DIR
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
fi

CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p $CHECKPOINT_DIR

CONFIG_BASE="cfg/vpl-gp/hhst.yaml"
TRAINER="vplgprewardchoicetrainer"
MODEL="qwen2"

CONFIG_FILE="cfg/hpsearch/vpl-gp-qwen/phase_${TID}.yaml"
mkdir -p $(dirname $CONFIG_FILE)

if [ ! -f "$CONFIG_BASE" ]; then
    echo "ERROR: Base config file not found: $CONFIG_BASE"
    exit 1
fi
cp $CONFIG_BASE $CONFIG_FILE

# Phase 1: 54100-54106 | Phase 2: 54107-54113 | Phase 3: 54114-54116 | Phase 4: 54117 | Phase 5: 54118-54138
ORTHOGONAL_WEIGHT=1.0
ORTHONORM_WEIGHT=0.1
PROTOTYPE_SCALE=5.0
KL_WEIGHT=0.1
GP_TEMPERATURE=1.0
LR=0.0001

if [ $TID -ge 54100 ] && [ $TID -le 54106 ]; then
    case $TID in
        54100) ORTHOGONAL_WEIGHT=0.2 ;;
        54101) ORTHOGONAL_WEIGHT=1.0 ;;
        54102) ORTHOGONAL_WEIGHT=5.0 ;;
        54103) ORTHONORM_WEIGHT=0.0 ;;
        54104) ORTHONORM_WEIGHT=0.5 ;;
        54105) PROTOTYPE_SCALE=2.0 ;;
        54106) PROTOTYPE_SCALE=10.0 ;;
    esac
fi

# Phase 2: VPL Core (54105 기반 — prototype_scale=2.0 사용)
if [ $TID -ge 54107 ] && [ $TID -le 54113 ]; then
    PROTOTYPE_SCALE=2.0
    case $TID in
        54107) KL_WEIGHT=0.02 ;;
        54108) KL_WEIGHT=0.05 ;;
        54109) KL_WEIGHT=0.1 ;;
        54110) KL_WEIGHT=0.2 ;;
        54111) GP_TEMPERATURE=0.5 ;;
        54112) GP_TEMPERATURE=2.0 ;;
        54113) GP_TEMPERATURE=5.0 ;;
    esac
fi

if [ $TID -ge 54114 ] && [ $TID -le 54116 ]; then
    case $TID in
        54114) LR=0.00005 ;;
        54115) LR=0.0001 ;;
        54116) LR=0.0002 ;;
    esac
fi

if [ $TID -ge 54118 ] && [ $TID -le 54138 ]; then
    if [ $TID -ge 54118 ] && [ $TID -le 54124 ]; then
        case $TID in
            54118) ORTHOGONAL_WEIGHT=0.1 ;;
            54119) ORTHOGONAL_WEIGHT=0.2 ;;
            54120) ORTHOGONAL_WEIGHT=0.5 ;;
            54121) ORTHOGONAL_WEIGHT=1.0 ;;
            54122) ORTHOGONAL_WEIGHT=2.0 ;;
            54123) ORTHOGONAL_WEIGHT=5.0 ;;
            54124) ORTHOGONAL_WEIGHT=10.0 ;;
        esac
    fi
    if [ $TID -ge 54125 ] && [ $TID -le 54130 ]; then
        case $TID in
            54125) ORTHONORM_WEIGHT=0.0 ;;
            54126) ORTHONORM_WEIGHT=0.05 ;;
            54127) ORTHONORM_WEIGHT=0.1 ;;
            54128) ORTHONORM_WEIGHT=0.2 ;;
            54129) ORTHONORM_WEIGHT=0.5 ;;
            54130) ORTHONORM_WEIGHT=1.0 ;;
        esac
    fi
    if [ $TID -ge 54131 ] && [ $TID -le 54137 ]; then
        case $TID in
            54131) KL_WEIGHT=0.01 ;;
            54132) KL_WEIGHT=0.02 ;;
            54133) KL_WEIGHT=0.05 ;;
            54134) KL_WEIGHT=0.1 ;;
            54135) KL_WEIGHT=0.2 ;;
            54136) KL_WEIGHT=0.5 ;;
            54137) KL_WEIGHT=1.0 ;;
        esac
    fi
fi

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
config['federate']['client_num'] = 10
config['federate']['sample_client_num'] = 5
config['federate']['save_to'] = "$CHECKPOINT_DIR/hhrl_choice_${MODEL}_fedbiscuit_u3_vplgp_ortho_t${TID}.ckpt"
config['data']['root'] = "$WORK_DIR/data"
config['model']['type'] = 'Qwen/Qwen2-0.5B@huggingface_llm'
config['trainer']['type'] = "$TRAINER"
config['expname'] = "vplgp_hhst_ortho_qwen_t${TID}"
config['train']['optimizer']['lr'] = float("$LR")
config['dataloader']['batch_size'] = 8
config['llm']['grad_accum_step'] = 4
config['llm']['vpl_use_gp_prior'] = True
config['llm']['vpl_latent_dim'] = 32
config['llm']['vpl_kl_weight'] = float("$KL_WEIGHT")
config['llm']['vpl_gp_temperature'] = float("$GP_TEMPERATURE")
config['llm']['vpl_feature_method'] = 'choice_logits'
config['llm']['vpl_use_feature_difference'] = True
config['llm']['vpl_use_difference_only'] = True
config['llm']['vpl_max_logvar'] = -3.0
config['llm']['vpl_orthogonal_weight'] = float("$ORTHOGONAL_WEIGHT")
config['llm']['vpl_orthogonal_orthonorm_weight'] = float("$ORTHONORM_WEIGHT")
config['llm']['vpl_use_manual_orthogonal_labels'] = True
config['llm']['vpl_num_prototypes'] = 2
config['llm']['vpl_prototype_scale'] = float("$PROTOTYPE_SCALE")
config['llm']['vpl_tsne_visualize_freq'] = 10
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
print(f"Config file created: {config_file}")
EOF

echo "Hyperparameters: orthogonal_weight=$ORTHOGONAL_WEIGHT orthonorm_weight=$ORTHONORM_WEIGHT prototype_scale=$PROTOTYPE_SCALE kl_weight=$KL_WEIGHT gp_temperature=$GP_TEMPERATURE lr=$LR"
echo "Starting Qwen selector training: TID=$TID"
python -u federatedscope/main.py --cfg $CONFIG_FILE > outputs/${TID}.log 2>&1
echo "Experiment completed: TID=$TID"
