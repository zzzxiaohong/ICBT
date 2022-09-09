from sys import maxsize
from weakref import ref
import numpy as np
import argparse

def read_file(file_name):
    with open(file_name, "r", encoding="UTF-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def load_score(file_name):
    lines = read_file(file_name)
    data = []
    for line in lines:
        line = [float(word) for word in line.split()]
        data.append(line)
    return data

def compute_score(data, sent_len):
    original_scores = np.transpose(np.array(data))
    means = [np.mean(s) for s in original_scores]
    vars = [np.var(s) for s in original_scores]
    scores = [(1-mean/var) for mean, var in zip(means, vars)]
    return means, vars, scores

def merge_word(sentence, scores, means, vars):
    final_sentence = []
    final_scores = []
    final_means = []
    final_vars = []
    now_word = ""
    now_score = 0.0
    now_means = 0.0
    now_vars = 0.0
    now_num = 0
    for word, score, mean, var in zip(sentence, scores, means, vars):
        # print(word, score)
        if word.endswith("@@"):
            now_word += word[:-2]
            now_score += score
            now_means += mean
            now_vars += var
            now_num += 1
        else:
            now_word += word
            now_score += score
            now_means += mean
            now_vars += var
            now_num += 1
            final_sentence.append(now_word)
            final_scores.append(now_score / now_num)
            final_means.append(now_means / now_num)
            final_vars.append(now_vars / now_num)
            now_word = ""
            now_score = 0.0
            now_means = 0.0
            now_vars = 0.0
            now_num = 0
    if now_word != "":
        final_sentence.append(now_word)
        final_scores.append(now_score)
        final_means.append(now_means / now_num)
        final_vars.append(now_vars / now_num)
    return final_sentence, final_scores, final_means, final_vars

def second_element(ele):
    return ele[1]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_file", required=True, type=str)
    # parser.add_argument("-n", "--dropout_num", required=True, type=int)
    # parser.add_argument("-o", "--output_file", required=False)
    # args = parser.parse_args()
    # dropout_num = args.dropout_num

    data = load_score("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/result/Sample.scores")
    target = read_file("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/result/Sample.target")
    reference = read_file("/data/zhanghongxiao/NMT/NN_Domain_mg/data/datamono/Education-train.bpe.mono.align.en")
    
    dropout_num = 20
    index = 0
    all_res = []
    trans_rate = {}
    while index + dropout_num < len(data):
        target_sent = target[index].split()
        ref_sent = reference[index//dropout_num].split()
        
        means, vars, scores = compute_score(data[index: index+dropout_num], len(data[index]))
        # print(len(target[index].split()), len(data[index]))
        target_sent, scores, means, vars = merge_word(target_sent, scores, means, vars)
        min_score = min(scores)
        max_score = max(scores)
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        sent_res = sorted(zip(target_sent, scores, means, vars, normalized_scores), key=second_element)
        all_res.append(sent_res)
        # for s, score, _, __ in sent:
        #     print("{}\t{:.4f}\t".format(s, score))
        # for s, score, mean, var in sent:
        #     print("{}\t{:>8.4f}\t{:.4f}\t{:.4f}".format(s, score, mean, var))
        # print('\n')
        target_sent = ' '.join(target_sent).replace("@@ ", "").split()
        ref_sent = ' '.join(ref_sent).replace("@@ ", "").split()
        for word in target_sent:
            if word in ref_sent: trans = True
            else: trans = False
            if word in trans_rate:
                trans_rate[word][1] += 1
                if trans: trans_rate[word][0] += 1
            else:
                if trans: trans_rate[word] = [1, 1]
                else: trans_rate[word] = [0, 1]
        index += dropout_num
    
    dict = {}
    for line in all_res:
        for word, score, mean, var, normalized_score in line:
            if word in dict:
                dict[word][0] += normalized_score
                dict[word][1] += 1
            else:
                dict[word] = [normalized_score, 1]
    res = []
    for key in dict:
        res.append([key, dict[key][0] / dict[key][1]])
    res = sorted(res, key=second_element)
    for line in res:
        assert line[0] in trans_rate
        rate = trans_rate[line[0]][0] / trans_rate[line[0]][1] * 100
        print("{:>10}\t{:.4f}\t{:>8.4f}%\t({:>3d}/{:>3d})".format(line[0], line[1], rate, trans_rate[line[0]][0], trans_rate[line[0]][1]))