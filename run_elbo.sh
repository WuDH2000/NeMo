SEED="${1:-42}"

# WANDB="092f5ae640f3e9fdef24532e5847609799854263" # replace with your own WandB API key

# MOUNTS='--container-mounts=/lustre:/lustre,/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_pretrain/root:/root'
# CONTAINER=/lustre/fsw/portfolios/llmservice/users/cchen1/containers/nemo_flow_jun.sqsh
BASE_DIR=/home/i-wudonghang/lat_rea
CODE_DIR=$BASE_DIR/NeMo
LHOTSE_DIR=/home/i-wudonghang/data/duplex_s2s/lhotse

CONFIG_PATH=$CODE_DIR/config
CONFIG_NAME="Qwen2.5-7B_elbo_logits"
LR=5e-5


PT_WEIGHT=0.96
TEXT_WEIGHT=0.0
QA_WEIGHT=0.04
T2T_LOSS=0.0



use_gated_fusion=false

audio_loss_weight=0.0

# EXP_NAME="${CONFIG_NAME}_LR${LR}_${TOTAL_NUM_GPUS}gpu_${LR}_Gate-${use_gated_fusion}_AudioLoss${audio_loss_weight}"
EXP_NAME="code_v2_lat_elbo_7b_3ltransformer_153_fixinferbug"
# EXP_NAME="code_v2_lat_elbo_7b_wothk_fixinferbug"
RESULTS_DIR=/mnt/donghang-jfs/duplex/exp/results/${EXP_NAME}
mkdir -p ${RESULTS_DIR}

# read -r -d '' cmd <<EOF
export PYTHONPATH="${CODE_DIR}:${LHOTSE_DIR}:${PYTHONPATH}" \
&& export HF_HOME="/mnt/donghang-jfs/pretrained_models" \
&& export TORCH_HOME="/mnt/donghang-jfs/pretrained_models" \
&& export NEMO_CACHE_DIR="/mnt/donghang-jfs/pretrained_models" \
&& export OMP_NUM_THREADS=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 \
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 ${CODE_DIR}/examples/speechlm2/s2s_duplex_speech_decoder_train_elbo_logits.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    exp_manager.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    data.train_ds.seed=$SEED \
    data.validation_ds.seed=$SEED \
    trainer.strategy.data_parallel_size=8 \
    model.optimizer.lr=${LR} \
    model.text_to_text_loss_weight=${T2T_LOSS} \
    model.use_gated_fusion=${use_gated_fusion} \
    model.audio_loss_weight=${audio_loss_weight}


