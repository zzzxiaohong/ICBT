work_dir= # your working directory path
tdom=$1
sl=zh
tl=en
iter_num=$2 # 1/2/3

## Apply lexical constraints on the translation outputs
trans_dir=$work_dir/output/checkpoints/pre-train/$tl-$sl/result/
    # The path of translation outputs file
    # for example:$work_dir/output/checkpoints/pre-train/$tl-$sl/result/
    # or $work_dir/output/checkpoints/ICBT/ICBT-conf/${tl}2${sl}/iter$((iter_num-1))/$tdom
dest_dir=$work_dir/data/dataicbt/ICBT-conf/zh2en/iter$iter_num/$tdom-tag
dict_path=$work_dir/output/dict/din
fname=${tdom}.ft.predict.tok
foutname=${tdom}-train.tok
num_works=10
mkdir -p $dest_dir

 # match
python $work_dir/match-iter-base.py \
    --fin_path $trans_dir --fin_name $fname\
    --fout_path $dest_dir --fout_name $foutname \
    --lan $sl \
    --fdict $dict_path/en-zh.dict \
    --pool_num $num_works \
    --max_num 3 

for i in `seq $((num_works-1))`
do 
    rm $trans_dir/$fname$i*
    rm $dest_dir/*$foutname$i*
done
 # replace
python $work_dir/replace.py --fpath $dest_dir --fname $foutname --lan $sl

## Apply lexical constraints on the valid dataset
valid_dir=$work_dir/data/din_data/$tdom/
fname=valid.tok.zh
foutname=$tdom-valid.tok
num_works=1
  # match
python $work_dir/match-iter-base.py \
    --fin_path $valid_dir --fin_name $fname\
    --fout_path $dest_dir --fout_name $foutname \
    --lan $sl \
    --fdict $dict_path/en-zh.dict \
    --pool_num $num_works \
    --max_num 3 

for i in `seq $((num_works-1))`
do 
    rm $valid_dir/$fname$i*
    rm $dest_dir/*$foutname$i*
done
 # replace
python $work_dir/replace.py --fpath $dest_dir --fname $foutname --lan $sl


## apply bpe
bpe_scripts=$work_dir/tools/subword-nmt 
bpe_model_dir=$work_dir/data/dout_data/bpe_model
python $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/${tdom}-train.tok.tag.$sl > $dest_dir/${tdom}-train.bpe.tag.$sl
python $bpe_scripts/apply_bpe.py -c $bpe_model_dir/enzh.bpe < $dest_dir/${tdom}-valid.tok.tag.$sl > $dest_dir/${tdom}-valid.bpe.tag.$sl


## merge the out-of-domain data
dout_tagdata_dir=$work_dir/data/datatag/dout
data_mono=$work_dir/data/datamono/mono-en
line_num=$(cat $dest_dir/$tdom-train.bpe.tag.$sl | wc -l)
python $work_dir/sample.py --sl $sl --tl $tl --num $line_num \
    --fin_name $dout_tagdata_dir/train.bpe.merge --fout_name $dest_dir/dout-$tdom-train.bpe.tag

cat $data_mono/$tdom-train.bpe.mono.$tl >> $dest_dir/dout-$tdom-train.bpe.tag.$tl
cat $dest_dir/$tdom-train.bpe.tag.$sl >> $dest_dir/dout-$tdom-train.bpe.tag.$sl

cat $valid_dir/valid.tok.en > $dest_dir/$tdom-valid.bpe.tag.$tl

