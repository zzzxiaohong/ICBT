work_dir= # your working directory path 
num_works=10
dom=$1

dest_dir=$work_dir/data/datatag/$dom/$dom-conf/ 
dict_path=$work_dir/output/dict/din
mkdir -p $dest_dir
sl=zh
tl=en

data_dir=$work_dir/data/datamono/mono-zh/
chs_word_dir=$work_dir/scripts/QE/output/CBT-conf/$dom.choosewords
fname=$dom-train.tok.mono
foutname=train.tok

# match
python match-confidence.py \
    --fin_path $data_dir --fin_name $fname\
    --fout_path $dest_dir --fout_name $foutname \
    --chs_word_dir $chs_word_dir \
    --lan zh \
    --fdict $dict_path/en-zh.dict \
    --pool_num $num_works 

for i in `seq $((num_works-1))`
do 
    rm $data_dir/$fname$i*
    rm $dest_dir/*$foutname$i*
done

# replace
python replace.py --fpath $dest_dir --fname $foutname --lan $sl

# apply bpe
bpe_scripts=$work_dir/tools/subword-nmt 
bpe_model_dir=$work_dir/data/dout_data/bpe_model
python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/train.tok.tag.$sl > $dest_dir/train.bpe.tag.$sl