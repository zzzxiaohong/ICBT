## Matching Sentences and Dictionaries in CBT-confidence Method
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from numpy import *

def write_file(list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in list:
            f.write(i + '\n')
def match(filepatch, fin_name, fchs_word_name, fout_name, src_lang, fdict_name, save_path, devide=' '):
    fsrc_name = filepatch + fin_name + '.' + src_lang
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(fchs_word_name, 'r', encoding='utf-8') as fchs_word,\
        open(fdict_name, 'r', encoding='utf-8') as  fdict:
        punc = set(["?", ",", ".", "!", "$", "%", "^", "&", "*", "@", "~", "`" "-", "+", "_", "=", "{", "}", "[", "]", "<", ">", "/", "'", '"', "(", ")", ":", ";" ])
        term_dict = list(set([term.strip() for term in fdict.readlines()]))
        term_dict = [pair for pair in term_dict if len(pair.split(devide)) > 1 and pair.split(devide)[0] not in punc and pair.split(devide)[1] not in punc]
        src_term_list, tgt_term_list = [], []
        for term in term_dict:
            src, tgt = term.split()
            src_term_list.append(tgt)
            tgt_term_list.append(src)
        matched_src_lines = []
        term_lines = []

        for src_line, chs_word_line in tqdm(zip(fsrc.readlines(), fchs_word.readlines())):
            src_line = src_line.strip()
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
            term_line.sort(key=lambda i: len(i), reverse=True)
            term_lines.append('\t'.join(term_line))

        fsrc_out_name = save_path + fout_name + '.match.' + src_lang
        write_file(matched_src_lines, fsrc_out_name)
        write_file(term_lines, save_path + fout_name + '.term_lines')

def run(args, i):
    fin_name = args.fin_name + str(i)
    fout_name = args.fout_name + str(i)
    match(args.fin_path, fin_name, args.chs_word_dir, fout_name, args.lan, args.fdict, args.fout_path)
def creat_batch(fin_name, src_lang, batch_num):
    fsrc_name = fin_name + '.' + src_lang
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc:
        fsrc_all_lines = fsrc.readlines()
        all_num = len(fsrc_all_lines)
        every_num = int(all_num / batch_num)
        for i in range(batch_num):
            start = every_num*i
            end = every_num * (i + 1)
            if i == batch_num - 1:
                end = all_num
            fsrc_batch_lines = fsrc_all_lines[start:end]
            fsrcout = open(fin_name + str(i+1) + '.' + src_lang, 'w', encoding='utf-8')
            for fsrc_line in fsrc_batch_lines:
                fsrcout.write(fsrc_line)

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
    parse.add_argument('--fin_path', default='', help="The path of input file.")
    parse.add_argument('--fin_name', default='', help="The name of input file.")
    parse.add_argument('--fout_path', default='', help="The path of output file.")
    parse.add_argument('--fout_name', default='', help="The name of output file.")
    parse.add_argument('--chs_word_dir', default='', help="The file name of the word chosen by QE.")
    parse.add_argument('--lan', default='en', help="The source language.")
    parse.add_argument('--fdict', default='', help="The file path of bilingual dictionary.")
    parse.add_argument('--pool_num', default=1, type=int, help="Number of processes.")
    args = parse.parse_args()

    creat_batch(args.fin_path+args.fin_name, args.lan, args.pool_num)

    p = Pool(args.pool_num)  
    results = []  
    for i in range(args.pool_num): 
        r = p.apply_async(run, args=(args, i+1,)) 
        results.append(r)  
    p.close() 
    p.join() 
    for i in results:
        i.get()
    merge(args.fout_path + args.fout_name, args.lan, args.pool_num)
    merge(args.fout_path + args.fout_name, batch_num=args.pool_num)


