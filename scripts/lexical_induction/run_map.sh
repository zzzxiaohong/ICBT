work_dir= # your working directory path 

embed_path=$work_dir/output/embed
out_path=$work_dir/output/dict/din
code_dir=$work_dir/tools/vecmap
mkdir -p $out_path
sl=en
tl=zh

CUDA_VISIBLE_DEVICES=3,4 nohup python3 $code_dir/map_embeddings.py \
--unsupervised \
--cuda \
$embed_path/all-train.mono.$sl.vec \
$embed_path/all-train.mono.$tl.vec \
$out_path/train.mapped.$sl.emb \
$out_path/train.mapped.$tl.emb \
--log $out_path/run_map.$vec_version.out \
> run_map.$vec_version.out &