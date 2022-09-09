import argparse
from multiprocessing import Pool
import os
from tqdm import tqdm
from numpy import *

def write_file(list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in list:
            f.write(i + '\n')

def _is_chinese_char(c):  
    if len(c) > 1:
        return any([_is_chinese_char(c_i) for c_i in c])
    cp = ord(c)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False
def match(filepatch, fin_name, fchs_word_name, fout_name, src_lang, tgt_lang, fdict_name, save_path, devide=' '):

    fsrc_name = filepatch + fin_name + '.' + src_lang
    ftgt_name = filepatch + fin_name + '.' + tgt_lang
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(ftgt_name, 'r', encoding='utf-8') as ftgt,\
        open(fchs_word_name, 'r', encoding='utf-8') as fchs_word,\
        open(fdict_name, 'r', encoding='utf-8') as  fdict:
        punc = set(["?", ",", ".", "!", "$", "%", "^", "&", "*", "@", "~", "`" "-", "+", "_", "=", "{", "}", "[", "]", "<", ">", "/", "'", '"', "(", ")", ":", ";" ])
        term_dict = list(set([term.strip() for term in fdict.readlines()]))
        term_dict = [pair for pair in term_dict if len(pair.split(devide)) > 1 and _is_chinese_char(pair.split(devide)[1])]
        src_term_list, tgt_term_list = [], []
        for term in term_dict:
            tgt, src = term.split()
            src_term_list.append(src)
            tgt_term_list.append(tgt)
        # print(len(src_term_list), src_term_list)
        match_list = []
        matched_src_lines, matched_tgt_lines = [], []
        term_lines = []
        match_num_list = []
        i = 0
        match_n = 0
        all_word_list = []
        print("len src_term_list:",len(src_term_list))
        for src_line, tgt_line, chs_word_line in tqdm(zip(fsrc.readlines(), ftgt.readlines(), fchs_word.readlines())):
            all_word_list.extend(src_line.strip().split())
            src_line, tgt_line = src_line.strip(), tgt_line.strip()
            chs_word_line = chs_word_line.strip().split()
            term_line = []
            for word in chs_word_line:
                if word in src_term_list:
                    idx = src_term_list.index(word)
                    src_word = word
                    tgt_word = tgt_term_list[idx]
                    term = src_word + ' ||| ' + tgt_word
                    term_line.append(term)
            matched_src_lines.append(src_line)
            matched_tgt_lines.append(tgt_line)
            term_line.sort(key=lambda i: len(i), reverse=True)
            term_lines.append('\t'.join(term_line))

        fsrc_out_name = save_path + fout_name + '.match.' + src_lang
        ftgt_out_name = save_path + fout_name + '.match.' + tgt_lang
        write_file(matched_src_lines, fsrc_out_name)
        write_file(matched_tgt_lines, ftgt_out_name)
        write_file(term_lines, save_path + fout_name + '.term_lines')

def run(args, i):
    fin_name = args.fin + str(i)
    fout_name = args.fout + str(i)
    match(args.file_path, fin_name, args.chs_word_dir, fout_name, \
        args.s, args.t, args.fdict, args.save_path)
def creat_batch(fin_name, src_lang, tgt_lang, batch_num):
    fsrc_name = fin_name + '.' + src_lang
    ftgt_name = fin_name + '.' + tgt_lang
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(ftgt_name, 'r', encoding='utf-8') as ftgt:   # + '.sample'
        fsrc_all_lines, ftgt_all_lines = fsrc.readlines(), ftgt.readlines()
        print(len(fsrc_all_lines), len(ftgt_all_lines))
        all_num = len(fsrc_all_lines)
        # all_num = 1019
        every_num = int(all_num / batch_num)
        for i in range(batch_num):
            start = every_num*i
            end = every_num * (i + 1)
            if i == batch_num - 1:
                end = all_num
            fsrc_batch_lines, ftgt_batch_lines = fsrc_all_lines[start:end], ftgt_all_lines[start:end]
            fsrcout = open(fin_name + str(i+1) + '.' + src_lang, 'w', encoding='utf-8')
            ftgtout = open(fin_name + str(i+1) + '.' + tgt_lang, 'w', encoding='utf-8')
            for fsrc_line, ftgt_line in zip(fsrc_batch_lines, ftgt_batch_lines):
                fsrcout.write(fsrc_line)
                ftgtout.write(ftgt_line)

def merge(file_name, lang = None, batch_num=10):
    if lang != None:
        fmerge_name = file_name + '.match.' + lang
        with open(fmerge_name, 'w', encoding='utf-8') as fout:
            for i in range(batch_num):
                fname = file_name + str(i+1) + '.match.' + lang
                with open(fname, 'r', encoding='utf-8') as fin:
                    for line in fin.readlines():
                        fout.write(line)
    else:
        with open(file_name + '.term_lines', 'w', encoding='utf-8') as fout:
            for i in range(batch_num):
                with open(file_name + str(i+1) + '.term_lines', 'r', encoding='utf-8') as fin:
                    for line in fin.readlines():
                        fout.write(line)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--file_path', default='./dataset/')
    parse.add_argument('--chs_word_dir', default='')
    parse.add_argument('--fin', default='test')
    parse.add_argument('--fout', default='tok')
    parse.add_argument('--save_path', default='./dataset/matched')
    parse.add_argument('--s', default='en')
    parse.add_argument('--t', default='de')
    parse.add_argument('--fdict', default='./dict/en-de.filter')
    parse.add_argument('--batch_num', default=1, type=int)
    args = parse.parse_args()

    creat_batch(args.file_path+args.fin, args.s, args.t, args.batch_num)

    p = Pool(args.batch_num)  
    results = []  
    for i in range(args.batch_num): 
        r = p.apply_async(run, args=(args, i+1,)) 
        results.append(r)  
    p.close() 
    p.join() 
    for i in results:
        print(i.get())
    merge(args.save_path + args.fout, args.s, args.batch_num)
    merge(args.save_path + args.fout, args.t, args.batch_num)
    merge(args.save_path + args.fout, batch_num=args.batch_num)


