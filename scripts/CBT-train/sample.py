import random
import argparse
def sample(fin_name, fout_name, sl, tl, re_num):
    with open("%s.%s"%(fin_name, sl), 'r', encoding='utf-8') as fsrc_in,\
        open("%s.%s"%(fin_name, tl), 'r', encoding='utf-8') as ftgt_in:
        lines = ["%s ||| %s"%(src_line.strip(), tgt_line.strip()) \
            for src_line, tgt_line in zip(fsrc_in.readlines(), ftgt_in.readlines())]
        lenth = len(lines)
        if lenth < re_num:
            tmp = re_num
            new_lines = lines[:]
            for i in range(tmp - lenth):
                idx = random.randint(0, lenth - 1)
                assert idx < lenth, "idx=" + str(idx)
                new_lines.append(lines[idx])
        else:
            new_lines = lines[:re_num] 
        random.shuffle(new_lines)
        assert len(new_lines) == re_num, "newlines=%s, renum=%s"%(len(new_lines), re_num)
        fsrc_out = open("%s.%s"%(fout_name, sl), 'w', encoding='utf-8')
        ftgt_out = open("%s.%s"%(fout_name, tl), 'w', encoding='utf-8')
        for line in new_lines:
            line = line.split(' ||| ')
            fsrc_out.write(line[0] + '\n')
            ftgt_out.write(line[1] + '\n')

parse = argparse.ArgumentParser()
parse.add_argument('--fin_name', default='', help="Name of input file.")
parse.add_argument('--fout_name', default='', help="Name of output file.")
parse.add_argument('--sl', default='en', help="The source language.")
parse.add_argument('--tl', default='zh', help="The target language.")
parse.add_argument('--num', default=0, type=int, help="The number after sample.")

args = parse.parse_args()

sample(args.fin_name, args.fout_name, args.sl, args.tl, args.num)
