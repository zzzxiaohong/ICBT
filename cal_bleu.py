import argparse
import sacrebleu
def cal_bleu(fsrc_name, fpre_name, ftgt_name):
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(fpre_name, 'r', encoding='utf-8') as fpre,\
        open(ftgt_name, 'r', encoding='utf-8') as ftgt:
        src_sentences = [line.strip() for line in fsrc.readlines()]
        prediction_sentences = [line.strip() for line in fpre.readlines()]
        golden_sentences = [line.strip() for line in ftgt.readlines()]
        print(ftgt_name + ':')
        bleu =sacrebleu.corpus_bleu(prediction_sentences,
                                    [src_sentences, golden_sentences], force=True)
        print(bleu)


parse = argparse.ArgumentParser()
parse.add_argument('--fsrc', default='', help="The name of the test set source file.")
parse.add_argument('--ftgt', default='', help="The name of the test set target reference file.")
parse.add_argument('--fpre', default='', help="The name of the translation output file for the test set.")
args = parse.parse_args()

cal_bleu(args.fsrc, args.fpre, args.ftgt)