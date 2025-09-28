export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'

model_name_or_path="tencent/Youtu-Embedding"
benchmark="MTEB(cmn, v1)"

python run_mteb.py \
  --model ${model_name_or_path} \
  --benchmark "${benchmark}" $@
