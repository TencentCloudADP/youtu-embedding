# 在 CMTEB 评测 embedding 模型
batch_size=512
max_length=1024

BASE_DIR=$BASE_DIR  # workspace 路径，需要根据实际情况修改

base_model_path=$BASE_DIR/model/bge-large-zh-v1.5
base_model_path=$BASE_DIR/model/multilingual-e5-large-instruct
base_model_path=$BASE_DIR/model/MiniCPM-Embedding
checkpoint_path=$BASE_DIR/output/0428_attn_1_pcc_cosent_not_cross/checkpoint-3824

python eval_cmteb.py \
    --base_model_path $base_model_path \
    --checkpoint_path $checkpoint_path \
    --use_task_instruction \
    --batch_size $batch_size \ 
    --max_length $max_length
