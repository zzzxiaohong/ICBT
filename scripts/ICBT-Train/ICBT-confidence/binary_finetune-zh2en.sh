work_dir= # your working directory path
tdom=$1
sl=zh
tl=en
iter_num=1 # 1/2/3

dict_path=$work_dir/output/data-bin-join/dout/$sl-$tl-tag
bin_path=$work_dir/output/data-bin-join/ICBT/ICBT-conf/zh2en/iter$iter_num/$tdom

mkdir -p $bin_path
data_path=$work_dir/data/dataicbt/ICBT-conf/zh2en/iter$iter_num/$tdom-tag

fairseq-preprocess -s $sl -t $tl \
  --srcdict $dict_path/dict.$sl.txt \
  --tgtdict $dict_path/dict.$tl.txt \
  --trainpref $data_path/dout-$tdom-train.bpe.tag \
  --validpref $data_path/$tdom-valid.bpe.tag \
  --workers 20 \
  --destdir $bin_path




