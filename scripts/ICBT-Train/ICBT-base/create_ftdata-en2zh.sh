work_dir= # your working directory path
tdom=$1
sl=en
tl=zh
iter_num=$2 # 1/2/3
trans_dir=$work_dir/output/checkpoints/ICBT/ICBT-base/zh2en/iter$iter_num/$tdom/result
dout_data_dir=$work_dir/data/dout_data
datamono_dir=$work_dir/data/datamono/mono-zh
dest_dir=$work_dir/data/dataicbt/ICBT-base/en2zh/iter$iter_num/$tdom
mkdir -p $dest_dir

# copy the translation output as the part of the source train data
cp $trans_dir/$tdom.ft.predict.tok $dest_dir/$tdom-train.tok.$sl

# apply bpe
bpe_scripts=$work_dir/tools/subword-nmt 
bpe_model_dir=$work_dir/data/dout_data/bpe_model
python $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/${tdom}-train.tok.$sl > $dest_dir/${tdom}-train.bpe.$sl

# merge the out-of-domain data
line_num=$(cat $dest_dir/$tdom-train.bpe.$sl | wc -l)
python $work_dir/sample.py --sl $sl --tl $tl --num $line_num \
    --fin_name $dout_data_dir/train.bpe --fout_name $dest_dir/dout-$tdom-train.bpe

cat $dest_dir/$tdom-train.bpe.$sl >> $dest_dir/dout-$tdom-train.bpe.$sl
cat $datamono_dir/$tdom-train.bpe.mono.$tl >> $dest_dir/dout-$tdom-train.bpe.$tl

