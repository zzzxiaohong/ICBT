import argparse
import random

def is_in(A, B):
    return any([A == B[i:i+len(A)] for i in range(0, len(B)-len(A)+1)])

def write_file(list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in list:
            f.write(i + '\n')

def replace_by_tag(filepath, filename, sl, tl):
   with open( "%s/%s.match.%s"%(filepath, filename, sl), 'r', encoding='utf-8') as fsrc,\
        open( "%s/%s.match.%s"%(filepath, filename, tl), 'r', encoding='utf-8') as ftgt,\
        open( "%s/%s.term_lines"%(filepath, filename), 'r', encoding='utf-8') as fterm_lines:
        src_lines = [line.strip() for line in fsrc.readlines()]
        tgt_lines = [line.strip() for line in ftgt.readlines()]
        term_lines = [line.strip() for line in fterm_lines.readlines()]
        
        src_res = list()
        tgt_res = list()
        for idx, (src_line, tgt_line, term_line) in enumerate(zip(src_lines, tgt_lines, term_lines)):
            if idx % 5000 == 0:
                print(idx)
            term_line = term_line.split('\t')
            if term_line == [['']] or term_line == ['']:
                pass
            else:
                term_line = [item.split(" ||| ") for item in term_line]
                src_line_re = src_line
                tgt_line_re = tgt_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line_re = src_line_re.replace(" %s "%(src_word), " %s "%(tgt_word))

                src_res.append(src_line_re)
                tgt_res.append(tgt_line_re)

        lines = list(zip(src_res, tgt_res))
        random.shuffle(lines)
        src_res[:], tgt_res[:] = zip(*lines)

        with open("%s/%s.tag.%s"%(filepath, filename, sl), "w", encoding="UTF-8") as f:
            for line in src_res: 
                f.write(line + '\n')
        with open("%s/%s.tag.%s"%(filepath, filename, tl), "w", encoding="UTF-8") as f:
            for line in tgt_res:
                f.write(line + '\n')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--fpath', default='', help="Path of input or output file.")
    parse.add_argument('--fname', default='', help="Name of input file.")
    parse.add_argument('--sl', default='en', help="The source language.")
    parse.add_argument('--tl', default='zh', help="The target language.")
    args = parse.parse_args()


    replace_by_tag(args.fpath, args.fname, args.sl, args.tl)
