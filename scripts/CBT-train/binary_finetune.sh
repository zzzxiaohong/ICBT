work_dir= # your working directory path
tdom=$1
sl=en
tl=zh
version=$2 # base/dspec/conf

bin_path=$work_dir/output/data-bin-join/CBT/CBT-$version/$tdom
train_data_path=$work_dir/data/datacbt/CBT-$version/$tdom
valid_data_path=$work_dir/data/din_data/$tdom
dict_path=$work_dir/output/data-bin-join/dout/$sl-$tl
mkdir -p $bin_path

fairseq-preprocess -s $sl -t $tl \
  --srcdict $dict_path/dict.$sl.txt \
  --tgtdict $dict_path/dict.$tl.txt \
  --trainpref $train_data_path/dout-$tdom-train.bpe \
  --validpref $valid_data_path/valid.bpe \
  --workers 20 \
  --destdir $bin_path
