work_dir= # your working directory path 

tdom=$1
sl=en
tl=zh
export CUDA_VISIBLE_DEVICES=$2

model_dir= # Model path for inference, for example: $work_dir/output/checkpoints/cbt/Laws/Laws-base-en-zh
data_dir=$work_dir/output/data-bin-join/test/dout-${tdom}
din_data_dir=$work_dir/data/din_data/$tdom
log_dir=$model_dir/log
result_dir=$model_dir/result
mkdir -p $log_dir $result_dir


fairseq-generate $data_dir \
    --path $model_dir/checkpoint_best.pt --batch-size 128 --remove-bpe \
    --gen-subset test --beam 5 > $log_dir/infer.$tdom.out 2>&1 

grep ^H $log_dir/infer.$tdom.out | sort -n -k 2 -t '-' | cut -f 3 > $result_dir/$tdom.predict.tok
sed 's/<unk>//g' -i $result_dir/$tdom.predict.tok

python $work_dir/cal_bleu.py --fsrc $din_data_dir/test.tok.$sl \
    --ftgt $din_data_dir/test.tok.$tl --fpre $result_dir/$tdom.predict.tok