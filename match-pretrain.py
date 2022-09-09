import argparse
from multiprocessing import Pool
from tqdm import tqdm
from numpy import *

def write_file(list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in list:
            f.write(i + '\n')
def match(filepatch, fin_name, fout_name, src_lang, tgt_lang, fdict_name, save_path, max_num):
    fsrc_name = filepatch + '/' + fin_name + '.' + src_lang
    ftgt_name = filepatch + '/' + fin_name + '.' + tgt_lang
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(ftgt_name, 'r', encoding='utf-8') as ftgt,\
        open(fdict_name, 'r', encoding='utf-8') as  fdict:
        punc = set(["?", ",", ".", "!", "$", "%", "^", "&", "*", "@", "~", "`" "-", "+", "_", "=", "{", "}", "[", "]", "<", ">", "/", "'", '"', "(", ")", ":", ";" ])
        term_dict = list(set([term.strip() for term in fdict.readlines()]))
        term_dict = [pair for pair in term_dict if len(pair.split('\t')) > 1 and pair.split('\t')[0] not in punc and pair.split('\t')[1] not in punc]
        src_term_list = [term.split('\t')[1] for term in term_dict]
        tgt_term_list = [term.split('\t')[0] for term in term_dict]
        matched_src_lines, matched_tgt_lines = [], []
        term_lines = []
        for src_line, tgt_line in tqdm(zip(fsrc.readlines(), ftgt.readlines())):
            src_line, tgt_line = src_line.strip(), tgt_line.strip()
            term_line = []
            for src_term, tgt_term in zip(src_term_list, tgt_term_list):
                if len(term_line) >= max_num:
                    break
                if src_term in src_line:
                    term_start_pos = src_line.index(src_term)
                    term_end_pos = term_start_pos + len(src_term)
                    if term_start_pos == 0:
                        new_src_term = src_term + ' '
                    elif term_end_pos == len(src_term):
                        new_src_term = ' ' + src_term
                    else:
                        new_src_term = ' ' + src_term + ' '
                    if new_src_term not in src_line:
                        continue
                    if tgt_term in tgt_line:
                        term = src_term + ' ||| ' + tgt_term
                        flag = True
                        for word in term_line:
                            if term in word:
                                flag = False
                        if term not in term_line and len(term_line) < max_num and flag == True:
                            term_line.append(term)
            matched_src_lines.append(src_line)
            matched_tgt_lines.append(tgt_line)
            term_line.sort(key=lambda i: len(i), reverse=True)
            term_lines.append('\t'.join(term_line))

        fsrc_out_name = save_path + '/' + fout_name + '.match.' + src_lang
        ftgt_out_name = save_path + '/' + fout_name + '.match.' + tgt_lang
        write_file(matched_src_lines, fsrc_out_name)
        write_file(matched_tgt_lines, ftgt_out_name)
        write_file(term_lines, save_path + '/' + fout_name + '.term_lines')

def run(args, i):
    fin_name = args.fin_name + str(i)
    fout_name = args.fout_name + str(i)
    match(args.fin_path, fin_name, fout_name, args.s, args.t, args.fdict, args.fout_path, args.max_num)
def creat_batch(fin_name, src_lang, tgt_lang, pool_num):
    fsrc_name = fin_name + '.' + src_lang
    ftgt_name = fin_name + '.' + tgt_lang
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(ftgt_name, 'r', encoding='utf-8') as ftgt:   # + '.sample'
        fsrc_all_lines, ftgt_all_lines = fsrc.readlines(), ftgt.readlines()
        all_num = len(fsrc_all_lines)
        every_num = int(all_num / pool_num)
        for i in range(pool_num):
            start = every_num*i
            end = every_num * (i + 1)
            if i == pool_num - 1:
                end = all_num
            fsrc_batch_lines, ftgt_batch_lines = fsrc_all_lines[start:end], ftgt_all_lines[start:end]
            fsrcout = open(fin_name + str(i+1) + '.' + src_lang, 'w', encoding='utf-8')
            ftgtout = open(fin_name + str(i+1) + '.' + tgt_lang, 'w', encoding='utf-8')
            for fsrc_line, ftgt_line in zip(fsrc_batch_lines, ftgt_batch_lines):
                fsrcout.write(fsrc_line)
                ftgtout.write(ftgt_line)

def merge(file_name, lang = None, pool_num=10):
    if lang != None:
        fmerge_name = file_name + '.match.' + lang
        with open(fmerge_name, 'w', encoding='utf-8') as fout:
            for i in range(pool_num):
                fname = file_name + str(i+1) + '.match.' + lang
                with open(fname, 'r', encoding='utf-8') as fin:
                    for line in fin.readlines():
                        fout.write(line)
    else:
        with open(file_name + '.term_lines', 'w', encoding='utf-8') as fout:
            for i in range(pool_num):
                with open(file_name + str(i+1) + '.term_lines', 'r', encoding='utf-8') as fin:
                    for line in fin.readlines():
                        fout.write(line)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--fin_path', default='', help="The path of input file.")
    parse.add_argument('--fin_name', default='', help="The name of input file.")
    parse.add_argument('--fout_path', default='', help="The path of output file.")
    parse.add_argument('--fout_name', default='', help="The name of output file.")
    parse.add_argument('--s', default='en', help="The source language.")
    parse.add_argument('--t', default='zh', help="The target language.")
    parse.add_argument('--fdict', default='', help="The file path of bilingual dictionary.")
    parse.add_argument('--pool_num', default=1, type=int, help="Number of processes.")
    parse.add_argument('--max_num', default=3, type=int, help="Maximum number of replacement words per sentence.")
    args = parse.parse_args()

    creat_batch(args.fin_path+'/'+args.fin_name, args.s, args.t, args.pool_num)

    p = Pool(args.pool_num)  
    results = []  
    for i in range(args.pool_num): 
        r = p.apply_async(run, args=(args, i+1,)) 
        results.append(r)  
    p.close() 
    p.join() 
    for i in results:
        i.get()
    merge(args.fout_path + '/' + args.fout_name, args.s, args.pool_num)
    merge(args.fout_path + '/' + args.fout_name, args.t, args.pool_num)
    merge(args.fout_path + '/' + args.fout_name, pool_num=args.pool_num)
