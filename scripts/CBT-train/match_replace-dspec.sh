work_dir= # your working directory path 
tdom=$1
cuda_id=$2
lan=zh
text_name=$work_dir/data/datamono/mono-zh/$tdom-train.tok.mono.zh
dictpath=$work_dir/output/dict/din/en-zh.dict
out_dir=$work_dir/data/datatag/$tdom/$tdom-dspec
log_dir=$work_dir/scripts/CBT-train/log
model_dir=$work_dir/output/language_model/$lan
mkdir -p $out_dir $log_dir

## 1 match
nohup python -u match-domainspec.py \
        --din_bert_dir $model_dir/$tdom \
        --dout_bert_dir $model_dir/dout \
        --textname $text_name \
        --dictname $dictpath \
        --outname $out_dir/train.tok \
        --batch_size 180 \
        --max_matched_num 3 \
        --device cuda:$cuda_id > $log_dir/match-dspec.$tdom.log &

# ## 2 replace
# python replace.py --fpath $out_dir --fname train.tok --lan $lan

# ## 3 apply bpe
# bpe_scripts=$work_dir/tools/subword-nmt 
# bpe_model_dir=$work_dir/data/dout_data/bpe_model
# python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $out_dir/train.tok.tag.$lan > $out_dir/train.bpe.tag.$lan