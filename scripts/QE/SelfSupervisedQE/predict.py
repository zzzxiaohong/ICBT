import argparse
import numpy as np
import torch

from data import eval_collate_fn, EvalDataset
from evaluate import predict, make_word_outputs_final
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    set_seed,
)
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--test-src', type=str, default="")
parser.add_argument('--test-tgt', type=str, default="")
parser.add_argument('--threshold-tune', type=str)
parser.add_argument('--threshold', type=str)

parser.add_argument('--block-size', type=int, default=512)
parser.add_argument('--wwm', action='store_true')
parser.add_argument('--predict-n', type=int, default=40)
parser.add_argument('--predict-m', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--mc-dropout', action='store_true')

parser.add_argument('--checkpoint', type=str, default="")

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--threshold-output', type=str)
parser.add_argument('--score-output', type=str, default="")
args = parser.parse_args()
print(args)

set_seed(args.seed)
device = torch.device('cuda')
torch.cuda.set_device(0)

config = AutoConfig.from_pretrained(args.checkpoint, cache_dir=None)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, cache_dir=None, use_fast=False, do_lower_case=False)

model = AutoModelWithLMHead.from_pretrained(args.checkpoint, config=config, cache_dir=None)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

test_dataset = EvalDataset(
    src_path=args.test_src,
    tgt_path=args.test_tgt,
    tokenizer=tokenizer,
    block_size=args.block_size,
    wwm=args.wwm,
    N=args.predict_n,
    M=args.predict_m,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=eval_collate_fn,
)

preds, preds_prob = predict(
    eval_dataloader=test_dataloader,
    model=model,
    device=device,
    tokenizer=tokenizer,
    N=args.predict_n,
    M=args.predict_m,
    mc_dropout=args.mc_dropout,
)

if args.threshold_tune:
    assert(args.threshold is None)
    word_scores, word_outputs, threshold, _ = make_word_outputs_final(preds, args.test_tgt, tokenizer, threshold_tune=args.threshold_tune)
    
    fth = open(args.threshold_output, 'w')
    fth.write(str(threshold))
    fth.close()
else:
    # assert(args.threshold_output is None)
    # fth = open(args.threshold, 'r')
    # threshold = float(fth.read().strip())
    # fth.close()
    
    word_scores, word_outputs, _, _ = make_word_outputs_final(preds, args.test_tgt, tokenizer, threshold=1.0)
for i in test_dataset.skip_ids: ##################add
    word_scores.insert(i, 'NAN')  #################add
fout_score = open(args.score_output, 'w')
for w in word_scores:
    fout_score.write(' '.join([str(x) for x in w]) + '\n')
fout_score.close()

print("Done.")

