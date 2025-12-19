#!/bin/bash
#SBATCH -A convai_convaird_nemo-speech
#SBATCH -J "s2s-train"
#SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4
#SBATCH -N 8 # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0



SEED="${1:-42}"

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
TOTAL_NUM_GPUS=`expr $GPUS_PER_NODE \* $SLURM_JOB_NUM_NODES`

WANDB="092f5ae640f3e9fdef24532e5847609799854263" # replace with your own WandB API key

MOUNTS='--container-mounts=/lustre:/lustre,/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_pretrain/root:/root'
CONTAINER=/lustre/fsw/portfolios/llmservice/users/cchen1/containers/nemo_flow_jun.sqsh
BASE_DIR=/lustre/fsw/portfolios/llmservice/users/cchen1/code/s2s_oct
CODE_DIR=$BASE_DIR/NeMo
LHOTSE_DIR=$BASE_DIR/lhotse

CONFIG_PATH=$BASE_DIR/conf/train
CONFIG_NAME="Qwen2.5-7B_PT_ParakeetEnc600m_semantic"
LR=5e-5


PT_WEIGHT=0.96
TEXT_WEIGHT=0.0
QA_WEIGHT=0.04
T2T_LOSS=0.0



use_gated_fusion=false

audio_loss_weight=0.0

EXP_NAME="${CONFIG_NAME}_LR${LR}_${TOTAL_NUM_GPUS}gpu_${LR}_Gate-${use_gated_fusion}_AudioLoss${audio_loss_weight}"
RESULTS_DIR=$BASE_DIR/exp_PT_semantic/${EXP_NAME}
mkdir -p ${RESULTS_DIR}

read -r -d '' cmd <<EOF
export WANDB_API_KEY="${WANDB}" \
&& export AIS_ENDPOINT="http://asr.iad.oci.aistore.nvidia.com:51080" \
&& export PYTHONPATH="${CODE_DIR}:${LHOTSE_DIR}:${PYTHONPATH}" \
&& export HF_HOME="/lustre/fsw/portfolios/llmservice/users/cchen1/hfcache" \
&& export TORCH_HOME="/lustre/fsw/portfolios/llmservice/users/cchen1/hfcache" \
&& export NEMO_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/cchen1/hfcache" \
&& export OMP_NUM_THREADS=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 \
python ${CODE_DIR}/examples/speechlm2/s2s_duplex_speech_decoder_train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    exp_manager.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    data.train_ds.seed=$SEED \
    data.validation_ds.seed=$SEED \
    trainer.strategy.data_parallel_size=${TOTAL_NUM_GPUS} \
    model.optimizer.lr=${LR} \
    data.train_ds.sampler_weights.interleave_s2s=${PT_WEIGHT} \
    data.train_ds.sampler_weights.single_turn=${QA_WEIGHT} \
    data.train_ds.sampler_weights.text_extract_knowledge=${TEXT_WEIGHT} \
    model.text_to_text_loss_weight=${T2T_LOSS} \
    model.use_gated_fusion=${use_gated_fusion} \
    model.audio_loss_weight=${audio_loss_weight}

EOF


#trainer.strategy.data_parallel_size=${TOTAL_NUM_GPUS} \

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/slurm-%j-%n.err

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
