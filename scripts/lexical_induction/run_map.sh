work_dir= # your working directory path 
embed_path=$work_dir/output/embed
out_path=$work_dir/output/dict/din
code_dir=$work_dir/tools/vecmap
mkdir -p $out_path
sl=en
tl=zh
gpu_ids=$1
CUDA_VISIBLE_DEVICES=gpu_ids python $code_dir/map_embeddings.py \
    --unsupervised \
    --cuda \
    $embed_path/all-train.mono.$sl.vec \
    $embed_path/all-train.mono.$tl.vec \
    $out_path/train.mapped.$sl.emb \
    $out_path/train.mapped.$tl.emb \
    --log $out_path/run_map.out