import argparse
import os
import numpy as np
import random

def is_in(A, B):
    return any([A == B[i:i+len(A)] for i in range(0, len(B)-len(A)+1)])

def write_file(list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in list:
            f.write(i + '\n')

def replace_by_tag(filepath, filename, lan):
    with open( "%s/%s.match.%s"%(filepath, filename, lan), 'r', encoding='utf-8') as fsrc,\
        open("%s/%s.term_lines"%(filepath, filename), 'r', encoding='utf-8') as fterm_lines:
        src_lines = [line.strip() for line in fsrc.readlines()]
        term_lines = [line.strip() for line in fterm_lines.readlines()]

        src_res=[]
        for src_line, term_line in zip(src_lines, term_lines):
            term_line = term_line.split('\t')
            if term_line == [''] or term_line == [['']]:
                src_res.append(src_line)
                pass
            else:
                term_line = [item.split(" ||| ") for item in term_line]
                src_line_re = src_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line_re = src_line_re.replace(" %s "%(src_word), " %s "%(tgt_word))              
                src_res.append(src_line_re)
        
        assert len(src_res) == len(src_lines),\
             "len(src_res):%d, len(src_lines):%d"%(len(src_res), len(src_lines))
        with open("%s/%s.tag.%s"%(filepath, filename, lan), "w", encoding="UTF-8") as f:
            for line in src_res:
                f.write(line + '\n')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--fpath', default='', help="Path of input or output file.")
    parse.add_argument('--fname', default='', help="Name of input file.")
    parse.add_argument('--lan', default='en', help="The source language.")
    args = parse.parse_args()

    replace_by_tag(args.fpath, args.fname, args.lan)