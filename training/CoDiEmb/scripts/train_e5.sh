#!/bin/bash
set -ex

BASE_DIR=$BASE_DIR  # workspace 路径，需要根据实际情况修改
IR_DATA_PATH=$BASE_DIR/CoDiEmb/data/ir_example_data
STS_DATA_PATH=$BASE_DIR/CoDiEmb/data/sts_example_data
export HF_DATASETS_CACHE=$IR_DATA_PATH/cache
export WANDB_DISABLED=true

model_name_or_path=$BASE_DIR/model/multilingual-e5-large-instruct
OUTPUT_PATH=$BASE_DIR/output/0804-e5-ins-ir64-sts32-p2n4-5epoch-lr5e-5

mkdir -p $OUTPUT_PATH
LOG_PATH=$OUTPUT_PATH
mkdir -p $LOG_PATH/logs

PROJECT_PATH=$BASE_DIR/CoDiEmb
DS_PATH=$PROJECT_PATH/scripts/deepspeed_config_fp32_zero1.json
cd $PROJECT_PATH

# 去除 deepspeed 并不会产生太多显存消耗
LAUNCHER="python3 -m torch.distributed.run \
    --nnodes $HOST_NUM \
    --node_rank $INDEX \
    --nproc_per_node $HOST_GPU_NUM \
    --master_addr $CHIEF_IP \
    --master_port 29500 \
    train/run.py \
    --deepspeed ${DS_PATH}"

# e5, bge 模型的 max_position_embeddings 为 512
# 在不开启 gradient_checkpointing 的情况下，bs 需要降到 16，且必须开启 bf16
export CMD=" \
    --output_dir ${OUTPUT_PATH} \
    --model_name_or_path ${model_name_or_path} \
    --cache_dir ${HF_DATASETS_CACHE} \
    --seed 49 \
    --ir_train_data ${IR_DATA_PATH} \
    --sts_train_data ${STS_DATA_PATH} \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --ir_per_device_batch_size 64 \
    --sts_per_device_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --ir_negatives_cross_device \
    --dataloader_drop_last \
    --data_sampler dynamic \
    --normalized \
    --temperature 0.02 \
    --multi_layer_loss \
    --positive_group_size 2 \
    --negative_group_size 4 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --logging_steps 10 \
    --pooling_method mean \
    --attn bbcc \
    --attn_implementation eager \
    --save_strategy "epoch" \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --skip_filter_too_long_instruction"
set +e

sh -c "$LAUNCHER $CMD" 2>&1 | tee $LOG_PATH/logs/${INDEX}.log
