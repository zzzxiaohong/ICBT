work_dir= # your working directory path 
export CUDA_VISIBLE_DEVICES=$2
# these values may be set larger for better performance
PREDICT_N=40
PREDICT_M=6
data_path=$work_dir/scripts/QE/output/CBT-conf
log_path=$work_dir/scripts/QE/logs
sl=zh
tl=en
dom=$1
src_data_name=$data_path/${dom}.source
tgt_data_name=$data_path/${dom}.target
model_dir=$work_dir/output/QE_model
code_dir=$work_dir/scripts/QE/SelfSupervisedQE
score_out=$data_path/${dom}.score.out
log_dir=$log_path/${dom}.predict.log
mkdir -p $log_path

## 1 calculate the scores of target words with QE model
python $code_dir/predict.py \
    --test-src=$src_data_name \
    --test-tgt=$tgt_data_name \
    --wwm \
    --mc-dropout \
    --predict-n=$PREDICT_N \
    --predict-m=$PREDICT_M \
    --checkpoint=$model_dir \
    --score-output=$score_out \
    --threshold=$log_path/threshold.out > $log_dir 2>&1

## 2 calculate the scores of source words with attention
python $code_dir/attn_with_qe.py --fpath $data_path --dom $dom 

## 3 choose the words with low confidence
python $code_dir/choose.py --fpath $data_path --dom $dom
