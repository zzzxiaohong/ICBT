work_dir= # your working directory path
export CUDA_VISIBLE_DEVICES=$2
tdom=$1
sl=zh
tl=en
model_dir=$work_dir/output/checkpoints/pre-train/$sl-$tl-tag
data_dir=$work_dir/data/datamono/mono-zh
dict_dir=$work_dir/output/data-bin-join/dout/$sl-$tl-tag
code_dir=$work_dir/tools/fairseq/fairseq_cli
result_dir=$work_dir/scripts/QE/output/CBT-conf
mkdir -p $result_dir

## 1 inference
cat $data_dir/$tdom-train.bpe.mono.zh | nohup python $code_dir/interactive.py  $dict_dir \
    --source-lang $sl --target-lang $tl \
    --path $model_dir/checkpoint_best.pt \
    --beam 5 \
    --batch-size 128 --remove-bpe \
    --buffer-size 800 \
    --unkpen 10 \
    > $result_dir/infer.$tdom.out 2>&1 & 
   
# ## 2 extract
# grep ^A $result_dir/infer.$tdom.out | cut -f2- > $result_dir/$tdom.attention
# grep ^S $result_dir/infer.$tdom.out | cut -f2- > $result_dir/$tdom.source
# grep ^H $result_dir/infer.$tdom.out | cut -f3- > $result_dir/$tdom.target