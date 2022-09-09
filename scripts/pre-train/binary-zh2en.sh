work_dir= # your working directory path
code_path=$work_dir/tools/fairseq/fairseq_cli
sl=zh
tl=en
data_path=$work_dir/data/datatag/dout
bin_path=$work_dir/output/data-bin-join/dout/$sl-$tl-tag

mkdir -p $bin_path


fairseq-preprocess -s $sl -t $tl \
  --srcdict $bin_path/dict.$sl.txt \
  --tgtdict $bin_path/dict.$tl.txt \
  --trainpref $data_path/train.bpe.merge \
	--validpref $data_path/valid.bpe.merge \
  --workers 20 \
  --destdir $bin_path