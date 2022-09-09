work_dir= # your working directory path
tdom=$1
cuda_id=$2
sl=zh
tl=en
iter_num=1 # 1/2/3

## 00 Apply lexical constraints on the translation outputs
trans_dir=$work_dir/output/checkpoints/pre-train/$tl-$sl/result/
    # The path of translation outputs file
    # for example:$work_dir/output/checkpoints/pre-train/$tl-$sl/result/
    # or $work_dir/output/checkpoints/ICBT/ICBT-dspec/${tl}2${sl}/iter$((iter_num-1))/$tdom
dest_dir=$work_dir/data/dataicbt/ICBT-dspec/zh2en/iter$iter_num/$tdom-tag
model_dir=$work_dir/output/language_model/$lan
dict_path=$work_dir/output/dict/din
fname=${tdom}.ft.predict.tok
foutname=${tdom}-train.tok
num_works=10
mkdir -p $dest_dir

 # 1 match 
nohup python -u $work_dir/match-domainspec.py \
        --din_bert_dir $model_dir/$tdom \
        --dout_bert_dir $model_dir/dout \
        --textname $trans_dir/$fname \
        --dictname $dict_path \
        --outname $dest_dir/$foutname\
        --batch_size 180 \
        --max_matched_num 3 \
        --device cuda:$cuda_id > $log_dir/match-iter$iter_num.train.$tdom.log &

#  # 2 replace
# python $work_dir/replace.py --fpath $dest_dir --fname $foutname --lan $sl

# ## 01 Apply lexical constraints on the valid dataset
# valid_dir=$work_dir/data/din_data/$tdom/
# fname=valid.tok.zh
# foutname=$tdom-valid.tok
# num_works=1
#   # 1 match
# nohup python -u $work_dir/match-domainspec.py \
#         --din_bert_dir $model_dir/$tdom \
#         --dout_bert_dir $model_dir/dout \
#         --textname $valid_dir/$fname \
#         --dictname $dict_path \
#         --outname $dest_dir/$foutname \
#         --batch_size 180 \
#         --max_matched_num 3 \
#         --device cuda:$cuda_id > $log_dir/match-iter$iter_num.valid.$tdom.log &
#  # 2 replace
# python $work_dir/replace.py --fpath $dest_dir --fname $foutname --lan $sl


# ## 02 apply bpe
# bpe_scripts=$work_dir/tools/subword-nmt 
# bpe_model_dir=$work_dir/data/dout_data/bpe_model
# python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/${tdom}-train.tok.tag.$sl > $dest_dir/${tdom}-train.bpe.tag.$sl
# python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/${tdom}-valid.tok.tag.$sl > $dest_dir/${tdom}-valid.bpe.tag.$sl


# ## 03 merge the out-of-domain data
# dout_tagdata_dir=$work_dir/data/datatag/dout
# data_mono=$work_dir/data/datamono/mono-en
# line_num=$(cat $dest_dir/$tdom-train.bpe.tag.$sl | wc -l)
# python $work_dir/sample.py --sl $sl --tl $tl --num $line_num \
#     --fin_name $dout_tagdata_dir/train.bpe.merge --fout_name $dest_dir/dout-$tdom-train.bpe.tag

# cat $data_mono/$tdom-train.bpe.mono.$tl >> $dest_dir/dout-$tdom-train.bpe.tag.$tl
# cat $dest_dir/$tdom-train.bpe.tag.$sl >> $dest_dir/dout-$tdom-train.bpe.tag.$sl

# cat $valid_dir/valid.tok.en > $dest_dir/$tdom-valid.bpe.tag.$tl

