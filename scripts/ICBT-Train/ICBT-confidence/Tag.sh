dom=$1

way=vecmap
version=v0

data_dir=/data/zhanghongxiao/NMT/NN_Domain_mg/data/dataiter-qe/zh-en-infer/three_iter/$dom-$way-${version}/

sl=zh
tl=en

python Tag.py --fpath $data_dir --fname $dom-train.tok --sl $sl --tl $tl --test
