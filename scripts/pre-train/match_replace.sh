work_dir= # your working directory path
num_works=10
dest_dir=$work_dir/data/datatag/dout
dict_path=$work_dir/output/dict/dout
mkdir -p $dest_dir
data_dir=$work_dir/data/dout_data

sl=zh
tl=en

# match
for split in 'train' 'valid'
do
    fname=$split.tok
    python match.py \
        --fin_path $data_dir --fin_name $fname\
        --fout_path $dest_dir --fout_name $fname \
        --s $sl --t $tl \
        --fdict $dict_path/en-zh.dict \
        --pool_num $num_works \
        --max_num 3 

    for i in `seq $((num_works-1))`
    do 
        rm $data_dir/$fname$i*
        rm $dest_dir/*$foutname$i*
    done
done

# replace
python replace.py --fpath $dest_dir --fname train.tok --sl $sl --tl $tl
python replace.py --fpath $dest_dir --fname valid.tok --sl $sl --tl $tl
cat $data_dir/train.tok.$sl $dest_dir/train.tok.tag.$sl > $dest_dir/train.tok.merge.$sl
cat $data_dir/train.tok.$tl $dest_dir/train.tok.tag.$tl > $dest_dir/train.tok.merge.$tl
cat $data_dir/valid.tok.$sl $dest_dir/valid.tok.tag.$sl > $dest_dir/valid.tok.merge.$sl
cat $data_dir/valid.tok.$tl $dest_dir/valid.tok.tag.$tl > $dest_dir/valid.tok.merge.$tl

# apply bpe
bpe_scripts=$work_dir/tools/subword-nmt 
bpe_model_dir=$work_dir/data/dout_data/bpe_model
python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/train.tok.merge.$sl > $dest_dir/train.bpe.merge.$sl
python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/train.tok.merge.$tl > $dest_dir/train.bpe.merge.$tl
python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/valid.tok.merge.$sl > $dest_dir/valid.bpe.merge.$sl
python3 $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/valid.tok.merge.$tl > $dest_dir/valid.bpe.merge.$tl