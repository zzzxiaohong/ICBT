work_dir= # your working directory path

export CUDA_VISIBLE_DEVICES=$2
sl=en
tl=zh
tdom=$1
version=base # base/dspec/conf

model_dir=$work_dir/output/checkpoints/pre-train/$sl-$tl
data_dir=$work_dir/output/data-bin-join/CBT/CBT-$version/$tdom
save_dir=$work_dir/output/checkpoints/CBT/CBT-$version/$tdom-$sl-$tl


epoch=40
log_dir=$save_dir/log
mkdir -p $save_dir $log_dir
nohup fairseq-train $data_dir \
              --save-dir $save_dir \
              --restore-file $model_dir/checkpoint_best.pt \
              --arch transformer \
              --source-lang ${sl} --target-lang ${tl} \
              --encoder-layers 6 --decoder-layers 6 \
              --encoder-embed-dim 512 --decoder-embed-dim 512 \
              --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
              --encoder-attention-heads 8 --decoder-attention-heads 8 \
              --encoder-normalize-before --decoder-normalize-before \
              --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
              --weight-decay 0.0001 \
              --eval-bleu --eval-bleu-args '{"beam": 5}' \
              --eval-tokenized-bleu --eval-bleu-remove-bpe \
              --best-checkpoint-metric bleu \
              --maximize-best-checkpoint-metric \
              --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
              --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
              --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
              --lr 5e-4 \
              --max-tokens 4096 \
              --update-freq 8 \
              --max-epoch ${epoch} --keep-interval-updates 10 --keep-last-epochs 1 \
              --fp16 \
              --reset-dataloader \
              --reset-optimizer \
              --save-interval-updates 5000 1> $log_dir/log.txt 2> $log_dir/err.txt &