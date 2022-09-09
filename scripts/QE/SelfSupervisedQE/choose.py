import heapq
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dom', type=str, default='')
parser.add_argument('--fpath', type=str, default='')

args = parser.parse_args()

def read_file(file_name):
    with open(file_name, "r", encoding="UTF-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

if __name__ == "__main__":
    fpath = args.fpath
    dom = args.dom
    source = read_file("{}/{}.source".format(fpath, dom))
    score = read_file("{}/{}.source.score".format(fpath, dom))

    choose_words = []

    for idx, (src, s) in enumerate(zip(source, score)):
        if idx % 10000 == 0:
            print(idx)
        src = src.split()
        if s == "N A N":
            choose_words.append(['', '', ''])
            continue
        s = [float(x) for x in s.split()]
        if not all(num == 0 for num in s):
            topk = heapq.nlargest(3, s)
        else:
            topk = random.sample(s, 3)
        topk_idx = [s.index(num) for num in topk]
        topk_words = [src[idx] for idx in topk_idx]
        
        choose_words.append(topk_words)

    with open("{}/{}.choosewords".format(fpath, dom), "w", encoding="UTF-8") as f:
        for line in choose_words:
            f.write('\t'.join(line) + '\n')
    
    print("Done.")
        