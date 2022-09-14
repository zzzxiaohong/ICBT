## creat pseudo-parallel data
work_dir= # your working directory path
sl=en
tl=zh
tdom=$1
version=$2 # base/dspec/conf
dout_data_dir=$work_dir/data/dout_data
din_data_dir=$work_dir/data/datamono/mono-zh
dest_dir=$work_dir/data/datacbt/CBT-$version/$tdom
trans_dir=$work_dir/output/checkpoints/pre-train/$tl-$sl-tag/result
bpe_model_dir=$dout_data_dir/bpe_model
bpe_scripts=$work_dir/tools/subword-nmt 
mkdir -p $dest_dir


cp $trans_dir/$tdom.$version.predict.tok $dest_dir/$tdom-train.tok.$sl
sed -i 's/<<unk>>\|<unk>//g' $dest_dir/$tdom-train.tok.$sl
## bpe
python $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/$tdom-train.tok.$sl > $dest_dir/$tdom-train.bpe.$sl
line_num=$(cat $dest_dir/$tdom-train.bpe.$sl | wc -l)

python $work_dir/sample.py --sl $sl --tl $tl --num $line_num \
    --fin_name $dout_data_dir/train.bpe --fout_name $dest_dir/dout-$tdom-train.bpe

cat $dest_dir/$tdom-train.bpe.$sl >> $dest_dir/dout-$tdom-train.bpe.$sl
cat $din_data_dir/$tdom-train.bpe.mono.$tl >> $dest_dir/dout-$tdom-train.bpe.$tl