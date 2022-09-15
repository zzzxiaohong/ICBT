work_dir= # your working directory path
export CUDA_VISIBLE_DEVICES=$2
tdom=$1
sl=zh
tl=en
iter_num=$3 # 1/2/3
model_dir=$work_dir/output/checkpoints/ICBT/ICBT-conf/zh2en/iter$iter_num/$tdom
data_dir=$work_dir/data/dataicbt/ICBT-conf/infer/iter$iter_num/$dom
dict_dir=$work_dir/output/data-bin-join/dout/$sl-$tl-tag
log_dir=$model_dir/log
result_dir=$model_dir/result
mkdir -p $log_dir $result_dir


## 1 inference
cat $data_dir/train.bpe.tag.zh | fairseq-interactive $dict_dir \
    --source-lang $sl --target-lang $tl \
    --path $model_dir/checkpoint_best.pt \
    --beam 5 \
    --batch-size 128 --remove-bpe \
    --buffer-size 800 \
    --unkpen 10 \
    > $log_dir/infer.$tdom.ft.out 2>&1


## 2 extract translation outputs
grep ^H $log_dir/infer.$tdom.ft.out | cut -f 3 > $result_dir/$tdom.ft.predict.tok
