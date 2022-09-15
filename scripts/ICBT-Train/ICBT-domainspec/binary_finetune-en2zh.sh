work_dir= # your working directory path
tdom=$1
sl=en
tl=zh
iter_num=$2 # 1/2/3
dict_dir=$work_dir/output/data-bin-join/dout/$sl-$tl
bin_dir=$work_dir/output/data-bin-join/ICBT/ICBT-dspec/en2zh/iter$iter_num/$tdom

mkdir -p $bin_dir
data_path=$work_dir/data/dataicbt/ICBT-dspec/en2zh/iter$iter_num/$tdom
valid_path=$work_dir/data/din_data/$tdom

fairseq-preprocess -s $sl -t $tl \
  --srcdict $dict_dir/dict.$sl.txt \
  --tgtdict $dict_dir/dict.$tl.txt \
  --trainpref $data_path/dout-$tdom-train.bpe \
  --validpref $valid_path/valid.bpe \
  --workers 20 \
  --destdir $bin_dir




