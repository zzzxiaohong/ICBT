work_dir= # your working directory path 
lan=zh
save_dir=$work_dir/output/language_model/$lan/dout
model_dir=$work_dir/scripts/MLM-train/bert-base-chinese
code_dir=$work_dir/tools/transformers/examples/pytorch/language-modeling
data_dir=$work_dir/data/dout_data
cp $data_dir/train.tok.$lan $data_dir/train.tok.$lan.txt
cp $data_dir/valid.tok.$lan $data_dir/valid.tok.$lan.txt
mkdir -p $save_dir

CUDA_VISIBLE_DEVICE=$1 python $code_dir/run_mlm.py \
    --model_name_or_path $model_dir \
    --train_file $data_dir/train.tok.$lan.txt \
    --validation_file $data_dir/valid.tok.$lan.txt \
    --max_seq_length 50 \
    --num_train_epochs 5 --save_steps 5000 --save_total_limit 1\
    --do_train --do_eval \
    --line_by_line \
    --fp16 \
    --overwrite_output_dir \
    --output_dir $save_dir \
    > ./log/dout.train.log
