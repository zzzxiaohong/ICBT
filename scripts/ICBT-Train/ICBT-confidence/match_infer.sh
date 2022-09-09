num_works=10
dom=$1
way=vecmap
version=v0
dest_dir=/data/zhanghongxiao/NMT/NN_Domain_mg/data/dataiter-qe/zh-en-infer/three_iter/$dom-$way-${version}/
dict_path=/data/zhanghongxiao/NMT/NN_Domain_mg/output/vecmap/$version
mkdir -p $dest_dir

fname=train.clean

data_dir=/data/zhanghongxiao/NMT/NN_Domain_mg/data/datamono/mono-zh/
chs_word_dir=/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/iter_data/iter3/output/$dom.choosewords
fname=$dom-train.tok.lc.mono.align

foutname=$dom-train.tok

python match_infer.py \
    --file_path $data_dir --fin ${fname} --fout $foutname\
    --chs_word_dir $chs_word_dir \
    --s zh --t en \
    --fdict $dict_path/en-zh.dict \
    --save_path $dest_dir\
    --batch_num $num_works 

for i in `seq $num_works`
do 
    rm $data_dir/$fname$i*
    rm $dest_dir/*$foutname$i*
done