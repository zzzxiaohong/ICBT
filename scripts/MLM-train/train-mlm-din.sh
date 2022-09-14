work_dir= # your working directory path 
lan=zh
dom=$1
save_dir=$work_dir/output/language_model/$lan/$dom
model_dir=$work_dir/scripts/MLM-train/bert-base-chinese
code_dir=$work_dir/tools/transformers/examples/pytorch/language-modeling
train_data_dir=$work_dir/data/datamono/mono-zh
valid_data_dir=$work_dir/data/din_data/$dom

cp $train_data_dir/$dom-train.tok.mono.$lan $train_data_dir/$dom-train.tok.mono.$lan.txt
cp $valid_data_dir/valid.tok.$lan $valid_data_dir/valid.tok.$lan.txt
mkdir -p $save_dir

CUDA_VISIBLE_DEVICE=$2 python $code_dir/run_mlm.py \
    --model_name_or_path $model_dir \
    --train_file $train_data_dir/$dom-train.tok.mono.$lan.txt \
    --validation_file $valid_data_dir/valid.tok.$lan.txt \
    --max_seq_length 50 \
    --num_train_epochs 5 --save_steps 5000 --save_total_limit 1\
    --do_train --do_eval \
    --line_by_line \
    --fp16 \
    --overwrite_output_dir \
    --output_dir $save_dir  \
    > ./log/$dom.train.log 
