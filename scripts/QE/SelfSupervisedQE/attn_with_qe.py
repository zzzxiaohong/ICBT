import argparse
import numpy as np

def read_file(file_name):
    with open(file_name, "r", encoding="UTF-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def reshape(attn, src_len, tgt_len, src_sent_len, tgt_sent_len):
    attn_mat = []
    for i in range(src_len):
        start_idx = i*tgt_len
        end_idx = start_idx + tgt_len
        attn_mat.append(attn[start_idx: end_idx])
    attn_mat = np.array(attn_mat)
    attn_mat = attn_mat[-1-src_sent_len:-1, :tgt_sent_len]
    
    return attn_mat.tolist()

def max_index(arr):
    mx = max(arr)
    return arr.index(mx)

def merge_bpe(src, tgt, attn):
    src_merge, tgt_merge = [], []

    attn = [list(map(float, attn_raw)) for attn_raw in attn]
    attn = np.array(attn)

    attn_merge = []
    now_word = ""
    now_score = np.zeros(len(attn[0]))
    now_len = 0
    idx = 0
    
    while idx < len(src):
        now_score += attn[idx]
        now_len += 1
        if src[idx].endswith("@@"):
            now_word += src[idx].replace("@@", "")
        else:
            now_word += src[idx]
            src_merge.append(now_word)
            attn_merge.append(now_score / now_len)
            now_word = ""
            now_score = np.zeros(len(attn[0]))
            now_len = 0
        idx += 1

    if len(now_word) > 0:
        src_merge.append(now_word)
        attn_merge.append(now_score)
    attn = np.array(attn_merge)
    
    attn_merge = []
    now_word = ""
    now_score = np.zeros(len(attn))
    now_len = 0
    idx = 0
    while idx < len(tgt):
        now_score += attn[:, idx]
        now_len += 1
        if tgt[idx].endswith("@@"):
            now_word += tgt[idx].replace("@@", "")
        else:
            now_word += tgt[idx]
            tgt_merge.append(now_word)
            attn_merge.append(now_score / now_len)
            now_word = ""
            now_score = np.zeros(len(attn))
            now_len = 0
        idx += 1
    if len(now_word) > 0:
        tgt_merge.append(now_word)
        attn_merge.append(now_score)
    attn = np.array(attn_merge)

    return src_merge, tgt_merge, attn.T
    
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)



parser = argparse.ArgumentParser()
parser.add_argument('--dom', type=str, default='')
parser.add_argument('--fpath', type=str, default='')
args = parser.parse_args()

if __name__ == "__main__":
    dom = args.dom
    fpath = args.fpath
    source = read_file("{}/{}.source".format(fpath, dom))
    target = read_file("{}/{}.target".format(fpath, dom))
    attention = read_file("{}/{}.attention".format(fpath, dom))
    scores = read_file("{}/{}.score.out".format(fpath, dom))

    src_score_all = []
    for idx, (src, tgt, attn, score) in enumerate(zip(source, target, attention, scores)):
        if idx % 1000 == 0:
            print("{} / {}".format(idx, len(source)))
        src = src.split()
        tgt = tgt.split()
        attn = attn.split('2022-0')[0]
        attn = np.array(eval(attn))
        attn = attn[-1-len(src):-1, -1-len(tgt):-1]
        if score == "N A N":
            src_score_all.append("NAN")
            continue
        else:
            score = np.array([float(s) for s in score.split()])
        if len(src) != attn.shape[0] or len(tgt) != attn.shape[1]:
            src_score_all.append([0 for i in range(len(src_merge))])
            continue
        src_merge, tgt_merge, attn_merge = merge_bpe(src, tgt, attn)
        if attn_merge.shape[1] != score.shape[0]:  
            src_score_all.append([0 for i in range(len(src_merge))])
            continue
        sent_score = attn_merge * score     
        src_score = np.sum(sent_score, axis=1)
        src_score_all.append(src_score)

    with open("{}/{}.source.score".format(fpath, dom), "w", encoding="UTF-8") as f:
        for line in src_score_all:
            f.write(' '.join([str(item) for item in line]) + '\n')
    print("{} Done.".format(dom))  