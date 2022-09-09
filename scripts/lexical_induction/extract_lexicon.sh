work_dir= # your working directory path 
data_dir=$work_dir/output/dict/din
sl=en
tl=zh

 CUDA_VISIBLE_DEVICES=6 nohup python3 $work_dir/extract_lexicon.py \
   --src_emb $data_dir/train.mapped.$sl.emb \
   --tgt_emb $data_dir/train.mapped.$tl.emb \
   --output $data_dir/$sl-$tl.dict \
   --dico_build "S2T&T2S" > $data_dir/extract_lexicon.out &