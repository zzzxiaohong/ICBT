work_dir= # your working directory path
export CUDA_VISIBLE_DEVICES=$2
tdom=$1
sl=en
tl=zh
iter_num=1 # 1/2/3
model_dir=$work_dir/output/checkpoints/pre-train/$sl-$tl
 # Model path for inference, for example: $work_dir/output/checkpoints/pre-train/$sl-$tl when starting iteration
 # or $work_dir/output/checkpoints/ICBT/ICBT-dspec/en2zh/iter$iter_num/$tdom
dict_dir=$work_dir/output/data-bin-join/dout/$sl-$tl
data_dir=$work_dir/data/datamono/mono-en
log_dir=$model_dir/log
result_dir=$model_dir/result
mkdir -p $result_dir $log_dir

## 1 inference
cat $data_dir/$tdom-train.bpe.mono.en.head | fairseq-interactive $dict_dir \
    --source-lang $sl --target-lang $tl \
    --path $model_dir/checkpoint_best.pt \
    --beam 5 \
    --batch-size 128 --remove-bpe \
    --buffer-size 800 \
    --unkpen 10 \
    > $log_dir/infer.$tdom.ft.out 2>&1


## 2 extract translation outputs
grep ^H $log_dir/infer.$tdom.ft.out | cut -f 3 > $result_dir/$tdom.ft.predict.tok