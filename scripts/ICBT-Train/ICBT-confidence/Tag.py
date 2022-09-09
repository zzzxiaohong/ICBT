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

def test_replace_by_tag(filepath, filename, sl, tl):
    with open( "%s/%s.match.%s"%(filepath, filename, sl), 'r', encoding='utf-8') as fsrc,\
        open( "%s/%s.match.%s"%(filepath, filename, tl), 'r', encoding='utf-8') as ftgt,\
        open("%s/%s.term_lines"%(filepath, filename), 'r', encoding='utf-8') as fterm_lines:
        # print('test')
        src_lines = [line.strip() for line in fsrc.readlines()]
        tgt_lines = [line.strip() for line in ftgt.readlines()]
        term_lines = [line.strip() for line in fterm_lines.readlines()]

        src_res = [list() for i in range(3)]
        # tgt_res = [list() for i in range(8)]
        tag_save = list()
        for idx, (src_line, term_line) in enumerate(zip(src_lines, term_lines)):
            # idx += 1
            term_line = term_line.split('\t')
            if term_line == [''] or term_line == [['']]:
                src_res[1].append(src_line)
                pass
            else:
                term_line = [item.split(" ||| ") for item in term_line]

                # method 1
                # ---------------------------------------------------------------------------------
                src_line1 = src_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line1 = src_line1.replace(" %s "%(src_word), " %s "%(tgt_word))

                

                src_res[1].append(src_line1)

        
        assert len(src_res[1]) == len(tgt_lines),\
             "len(src_res[1]):%d, len(tgt_lines):%d"%(len(src_res[1]), len(tgt_lines))

        for i in range(1, 2):
            with open("%s/%s.tag%s.%s"%(filepath, filename, i, sl), "w", encoding="UTF-8") as f:
                for line in src_res[i]:
                    f.write(line + '\n')
            with open("%s/%s.tag%s.%s"%(filepath, filename, i, tl), "w", encoding="UTF-8") as f:
                for line in tgt_lines:
                    f.write(line + '\n')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--fpath', default='./dataset/test.de')
    parse.add_argument('--fname', default='./dataset/test.en')
    parse.add_argument('--sl', default='en')
    parse.add_argument('--tl', default='de')
    parse.add_argument('--test', action="store_true")
    parse.add_argument('--train', action="store_true")
    args = parse.parse_args()



    if args.test:
        test_replace_by_tag(args.fpath, args.fname, args.sl, args.tl)
