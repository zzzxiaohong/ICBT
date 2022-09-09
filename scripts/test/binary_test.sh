work_dir= # your working directory path 
code_path=$work_dir/tools/fairseq/fairseq_cli
tdom=$1
sl=en
tl=zh
dict_path=$work_dir/output/data-bin-join/dout/$sl-$tl
bin_path=$work_dir/output/data-bin-join/test/dout-${tdom}
data_path=$work_dir/data/din_data/$tdom
mkdir -p $bin_path

fairseq-preprocess -s $sl -t $tl \
  --srcdict $dict_path/dict.$sl.txt \
  --tgtdict $dict_path/dict.$tl.txt \
  --testpref $data_path/test.bpe \
  --workers 20 \
  --destdir $bin_path

