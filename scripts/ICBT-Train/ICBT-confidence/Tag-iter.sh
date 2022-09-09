dom=$1
way=vecmap
max_match_num=3
version=v0_num$max_match_num

orgdata_dir=/data/zhanghongxiao/NMT/NN_Domain_mg/data/datamono/mono-en
data_dir=/data/zhanghongxiao/NMT/NN_Domain_mg/data/dataiter-qe/zh2en/three_iter/$dom-$way-$version
sl=zh
tl=en
cp $orgdata_dir/$dom-train.tok.lc.mono.$tl $data_dir/$dom-train.tok.tag1.$tl
python Tag-iter.py --fpath $data_dir --fname $dom-train.tok --sl $sl --tl $tl --test

orgdata_dir=/data/zhanghongxiao/NMT/Domain_mg/data/Domain_data/$dom
data_dir=/data/zhanghongxiao/NMT/NN_Domain_mg/data/dataiter-qe/zh2en/three_iter/$dom-$way-$version
sl=zh
tl=en
cp $orgdata_dir/valid.lc.tok.$tl $data_dir/$dom-valid.tok.tag1.$tl
python Tag-iter.py --fpath $data_dir --fname $dom-valid.tok --sl $sl --tl $tl --test