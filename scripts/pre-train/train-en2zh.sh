work_dir=/data/zhanghongxiao/NMT/ICBT # your working directory path

export CUDA_VISIBLE_DEVICES=$1 
sl=en
tl=zh
data_dir=$work_dir/output/data-bin-join/dout/$sl-$tl
code_path=$work_dir/tools/fairseq/fairseq_cli

epoch=40
save_dir=$work_dir/output/checkpoints/pre-train/${sl}-${tl}/
log_path=$save_dir/log
mkdir -p $log_path

nohup python $code_path/train.py $data_dir \
              --save-dir $save_dir \
              --arch transformer \
              --source-lang ${sl} --target-lang ${tl} \
              --encoder-layers 6 --decoder-layers 6 \
              --encoder-embed-dim 512 --decoder-embed-dim 512 \
              --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
              --encoder-attention-heads 8 --decoder-attention-heads 8 \
              --encoder-normalize-before --decoder-normalize-before \
              --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
              --weight-decay 0.0001 \
              --eval-bleu --eval-bleu-args '{"beam": 5}'  --maximize-best-checkpoint-metric \
              --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
              --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
              --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
              --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
              --lr 1e-3  --stop-min-lr 1e-9\
              --max-tokens 4096 --skip-invalid-size-inputs-valid-test\
              --update-freq 8 --share-all-embeddings \
              --max-epoch ${epoch} --keep-interval-updates 10 --keep-last-epochs 5 \
              --fp16 \
              --save-interval-updates 5000 1> $log_path/log.txt 2> $log_path/err.txt &