work_dir= # your working directory path
code_path=$work_dir/tools/fairseq/fairseq_cli
sl=en
tl=zh
data_dir=$work_dir/data/dout_data
bin_dir=$work_dir/output/data-bin-join/dout/$sl-$tl
mkdir -p $bin_dir


fairseq-preprocess -s $sl -t $tl \
  --srcdict $bin_dir/dict.$sl.txt \
  --tgtdict $bin_dir/dict.$tl.txt \
  --trainpref $data_dir/train.bpe \
  --validpref $data_dir/valid.bpe \
  --workers 20 \
  --destdir $bin_dir
