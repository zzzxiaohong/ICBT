work_dir= # your working directory path 
export CUDA_VISIBLE_DEVICES=$1
sl=zh
tl=en

data_path=$work_dir/data/dout_data
src_data_name=$data_path/train.tok.$sl
tgt_data_name=$data_path/train.tok.$tl
model_path=$work_dir/scripts/QE/SelfSupervisedQE/bert-base-multilingual-uncased
save_path=$work_dir/output/QE_model
log_path=$work_dir/scripts/QE/logs
code_path=$work_dir/scripts/QE/SelfSupervisedQE
mkdir -p $log_path

python $code_path/train.py \
    --train-src=$src_data_name \
    --train-tgt=$tgt_data_name \
    --wwm \
    --pretrained-model-path=$model_path \
    --save-model-path=$save_path \
    > $log_path/train.log 2>&1
