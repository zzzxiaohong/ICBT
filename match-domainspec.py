## Matching Sentences and Dictionaries in CBT-domainspec Method
from collections import defaultdict
import torch
from torch.utils import data
import argparse
import numpy as np
from transformers import BertForMaskedLM
from transformers import  BertTokenizer
from multiprocessing import Pool
from jieba.__init__ import Tokenizer


host = torch.device("cpu")
new_matched_lines = []
new_matched_probs = []
now_idx = 0
all_lins_num = 0
def readdict(filename, puns):
    dict_src, dict_tgt = [], []
    with open(filename, 'r', encoding="utf-8") as fin:
        for line in fin.readlines():
            line = line.strip().split()
            if len(line) > 1 and not (line[0].isdigit()) and not (line[1].isdigit()):
                if any(pun in line[0] for pun in puns) or any(pun in line[1] for pun in puns):
                    continue
                if not _is_chinese_char(line[1]):
                    continue
                dict_src.append(line[0])
                dict_tgt.append(line[1])
            else:
                continue
    return dict_src, dict_tgt

def readfile(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        lines = fin.readlines() #[:1000]
    global all_lins_num
    all_lins_num = len(lines)
    return lines

def text_to_id(token_text, word, tokenizer):
    input_ids= []
    for text in token_text:
        input_ids.append(tokenizer.convert_tokens_to_ids(text))
    word_id =  tokenizer.convert_tokens_to_ids(word)
    return input_ids, word_id

def align_linear(atokens, btokens):
    a2c = []
    c2b = []
    a2b = []
    length = 0
    for tok in atokens:
        a2c.append([length + i for i in range(len(tok))])
        length += len(tok)
    for i, tok in enumerate(btokens):
        c2b.extend([i for _ in range(len(tok))])
    for i, amap in enumerate(a2c):
        bmap = [c2b[ci] for ci in amap]
        a2b.append(list(set(bmap)))
    return a2b

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

def matched(lines, dict_tgt, dict_src, src_tokenizer, tgt_tokenizer, num, i):
    src_token_texts, tgt_token_texts = [], []
    src_mask_ids, tgt_mask_ids = [], []
    src_word_ids, tgt_word_ids = [], []
    every_num = int(len(lines) / num)
    if i == num-1:
        cur_lines = lines[i * every_num:]
    else:
        cur_lines = lines[i * every_num:(i+1) * every_num]
    matched_lines = []
    t = Tokenizer()
    for idx, line in enumerate(cur_lines):
        if idx % 1000 == 0:
            print("matched idx:", idx)    
        line = line.strip() 
        src_token_text = src_tokenizer.tokenize(line) 
        tgt_token_text = tgt_tokenizer.tokenize(line)
        src_token_text = ["[CLS]"] + src_token_text + ["[SEP]"]
        tgt_token_text = ["[CLS]"] + tgt_token_text + ["[SEP]"]
        
        src_tmp_line = [word.strip("##") for word in src_token_text] 
        tgt_tmp_line = [word.strip("##") for word in tgt_token_text]
        new_line = [src_tmp_line[0]] +  ' '.join(t.cut(''.join(src_tmp_line[1:-1]))).split() + [src_tmp_line[-1]]

        src_token_to_jie = align_linear(src_tmp_line, new_line) 
        tgt_token_to_jie = align_linear(tgt_tmp_line, new_line)

        for tgt_word, src_word in zip(dict_tgt, dict_src):
            tmp_src_token_text, tmp_tgt_token_text = src_token_text[:], tgt_token_text[:]
            if tgt_word in new_line:
                id = new_line.index(tgt_word)
                src_mask_id, tgt_mask_id = [], []
                src_matched_word, tgt_matched_word = [], []
                for i_num, idxs in enumerate(src_token_to_jie):
                    if id in idxs:
                        if i_num < 100:
                            src_matched_word.append(tmp_src_token_text[i_num])
                            tmp_src_token_text[i_num] = "[MASK]"                      
                            src_mask_id.append(i_num)
                        
                for i_num, idxs in enumerate(tgt_token_to_jie):
                    if id in idxs:
                        if i_num < 100:
                            tgt_matched_word.append(tmp_tgt_token_text[i_num])
                            tmp_tgt_token_text[i_num] = "[MASK]"
                            tgt_mask_id.append(i_num)

                src_word_id =  src_tokenizer.convert_tokens_to_ids(src_matched_word)
                tgt_word_id =  tgt_tokenizer.convert_tokens_to_ids(tgt_matched_word)
                src_token_texts.append(tmp_src_token_text)
                tgt_token_texts.append(tmp_tgt_token_text)
                src_mask_ids.append(src_mask_id)
                tgt_mask_ids.append(tgt_mask_id)
                src_word_ids.append(src_word_id)
                tgt_word_ids.append(tgt_word_id)
                matched_lines.append(str(i * every_num + idx) + '\t' + tgt_word + " ||| " + src_word)
            else:
                pass
    return src_token_texts, tgt_token_texts, src_mask_ids, tgt_mask_ids, src_word_ids, tgt_word_ids, matched_lines

def process_input(token_texts, tokenizer, max_seq_len=100):
    new_input_ids = []
    attention_masks = []
    for seq in token_texts:
        if len(seq) > max_seq_len:
            seq = seq[0 : max_seq_len]
        attention_mask = [1 for i in range(len(seq))]
        padding = [0] * (max_seq_len - len(seq))  
        seq = tokenizer.convert_tokens_to_ids(seq)
        seq += padding  
        attention_mask += padding
        assert len(seq) == max_seq_len
        assert len(attention_mask) == max_seq_len
        new_input_ids.append(seq)
        attention_masks.append(attention_mask)
    return new_input_ids, attention_masks

def creat_dataloader(input_ids, attention_masks, batch_size, shuffle = False):
    new_input_ids = torch.LongTensor(input_ids)
    attention_masks = torch.LongTensor(attention_masks)
    data_set = data.TensorDataset(new_input_ids, attention_masks)
    dataloder = data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloder
    
def predict(src_dataloader, tgt_dataloader, src_model, tgt_model, device, src_mask_ids, \
    tgt_mask_ids, src_word_ids, tgt_word_ids, matched_lines, src_tokenizer, tgt_tokenizer, all_batch_idxs):
    with torch.no_grad():
        src_model, tgt_model = src_model.to(device), tgt_model.to(device)
        idx = 0
        
        for [src_batch_input_ids, src_batch_attention_masks], [tgt_batch_input_ids, tgt_batch_attention_masks] in zip(src_dataloader, tgt_dataloader):
            src_preds, tgt_preds = [], []
            src_model.eval()
            tgt_model.eval()
            if (idx % 50 == 0):
                print("batch_idx: %d / %d"%(idx, all_batch_idxs))
            idx += 1
            src_batch_input_ids, tgt_batch_input_ids = src_batch_input_ids.to(device), tgt_batch_input_ids.to(device) #[0]
            src_batch_attention_masks, tgt_batch_attention_masks = \
                src_batch_attention_masks.to(device), tgt_batch_attention_masks.to(device)
            src_batch_outs, tgt_batch_outs = src_model(src_batch_input_ids, src_batch_attention_masks)[0],\
                 tgt_model(tgt_batch_input_ids, tgt_batch_attention_masks)[0]
            src_batch_outs_c, tgt_batch_outs_c = src_batch_outs.to(host), tgt_batch_outs.to(host)
            src_sam = [src_out.detach().numpy() for src_out in src_batch_outs_c]
            tgt_sam = [tgt_out.detach().numpy() for tgt_out in tgt_batch_outs_c]
            calculate(src_sam, tgt_sam, src_mask_ids, tgt_mask_ids, \
                src_word_ids, tgt_word_ids, matched_lines, src_tokenizer, tgt_tokenizer)
    return src_preds, tgt_preds

def calculate(src_preds, tgt_preds, src_mask_ids, tgt_mask_ids, src_word_ids, \
    tgt_word_ids, matched_lines, src_tokenizer, tgt_tokenizer):
    global now_idx 
    global new_matched_lines
    global new_matched_probs
    for src_line, tgt_line in zip(src_preds, tgt_preds):
        src_mask_id_line, tgt_mask_id_line = src_mask_ids[now_idx], tgt_mask_ids[now_idx]
        src_word_id_line, tgt_word_id_line = src_word_ids[now_idx], tgt_word_ids[now_idx]     
        src_word_pro, tgt_word_pro = [], []
        assert len(src_mask_id_line) == len(src_word_id_line), \
            f"len(src_mask_id_line) is {len(src_mask_id_line)}, len(src_word_id_line) is {len(src_word_id_line)}"
        for mask_id, word_id in zip(src_mask_id_line, src_word_id_line):
            src_word_pro.append(src_line[mask_id][word_id])
        assert len(tgt_mask_id_line) == len(tgt_word_id_line), \
            f"len(tgt_mask_id_line) is {len(tgt_mask_id_line)}, len(tgt_word_id_line) is {len(tgt_word_id_line)}"
        for mask_id, word_id in zip(tgt_mask_id_line, tgt_word_id_line):
            tgt_word_pro.append(tgt_line[mask_id][word_id])
        src_word_pro_avg, tgt_word_pro_avg = np.mean(src_word_pro), np.mean(tgt_word_pro) 
        if abs(src_word_pro_avg - tgt_word_pro_avg) > 6:
            new_matched_lines.append(matched_lines[now_idx])
            new_matched_probs.append(abs(src_word_pro_avg - tgt_word_pro_avg))
        else:
            pass
        now_idx += 1
    return 
def transfer(max_num):
    global new_matched_lines
    global new_matched_probs
    global all_lins_num
    final_lines = []
    lines_dict, ch_lines_dict = defaultdict(list), defaultdict(list)
    for line, prob in zip(new_matched_lines, new_matched_probs):
        line = line.split('\t')
        assert len(line) == 2,"len line = {}".format(len(line))
        lines_dict[int(line[0])].append(line[1] + '\t' + str(prob))
    for idx , line in lines_dict.items():
        matched = [word.split('\t')[0] for word in line]
        probs = [float(word.split('\t')[1]) for word in line]
        probs, matched = zip(*sorted(zip(probs, matched), reverse=True))
        ch_lines_dict[idx] = matched[:max_num]
    num = 0
    for idx, matched in ch_lines_dict.items():
        if idx == num:
            final_lines.append('\t'.join(matched))
        else:
            while num < idx:
                final_lines.append("")
                num +=1
            final_lines.append('\t'.join(matched))
        num += 1
    while num < all_lins_num:
        final_lines.append("")
        num += 1
    return final_lines


def writefile(lines, matched_lines, filename):
    assert len(lines) == len(matched_lines),"len lines = {}, len matched_lines = {}".format(len(lines), len(matched_lines))
    with open(filename + ".match.zh", 'w', encoding='utf-8') as fout,\
        open(filename + '.term_lines', 'w', encoding='utf-8') as fterms:
        for line, matched_words in zip(lines, matched_lines):
            fout.write(line)
            fterms.write(matched_words + '\n')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--din_bert_dir', default='', help="Path of the in-domain masked language model.")
    parse.add_argument('--dout_bert_dir', default='', help="Path of the out-of-domain masked language model.")
    parse.add_argument('--textname', default='', help="Filename of monolingual sentences that needs to be matched.")
    parse.add_argument('--dictname', default='', help="Filename of dictionary that needs to be matched.")
    parse.add_argument('--outname', default='', help="Filename of the output file.")
    parse.add_argument('--pool_num', type=int, default=40, help="Number of processes.")
    parse.add_argument('--batch_size', type=int, default=256, help="Size of mini-batch.")
    parse.add_argument('--device', default='cuda:0', help="The GPU ID.")
    parse.add_argument('--max_matched_num', default=3, type=int, help="Maximum number of replacement words per sentence.")
    args = parse.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    zh_puncs = ['，', '。', '？', '！', '；', '、', '：', '；', '‘', '’', '“', '”', '—', '——', '～','℃',
                '（', '）', '＝', '＞', '＜', '．', '／', '－', '＇', '＆', '％', '＄', '＃', '＂', '', '〔','〕', 
                '【', '】', '《', '》', '△', '■', '─', '▪', '▼', '★', '〃', '⑺', '⑹', '⑵', '④', '⑤', '⑦', '③', '②',
                '①', '≥', '≠', '∶', '≈', '∧', '″', '′', '―', 'ü', 'ā', 'ō', 'ú', 'ù', 'í', 'ì', 'ê', 'é', 'è', 'á', 'à', 
                '´', '»', '』', '╱', ""]
    en_puncs = [',', '.', '?', '!', ';', ':', ';',  '"', '-', '#', '/', '–', '−', '%', '$', '�',
			 '●', '•', '·', '≤', '(', ')', '€', '…', '@', '[', ']', '˚', '„', '‒', '‑', '*', '&', '+','}', '=',
             '~', '°', '±', '→', '￡', '１', '∞', '∝', '∑', '∈', 'Ⅷ', 'Ⅶ', 'Ⅵ', 'Ⅴ', 'Ⅳ', 'Ⅲ', 'Ⅱ', 'Ⅰ', 
             '™', '℉', '×', 'кB', 'ЛЧ', 'ωf', 'ω', 'χ', 'φn', 'σ', 'τ', '_', 'μ', 'λ', 'κ', 'θ', 'η', 'δ', 'ε', 'γ',
             'α', 'β', 'Φ', 'ˉ', 'Δ', 'Ω', 'Â', '®', '¬', '«', '¨', '§', '{', '￠', '＊', '［', '］', '￣', '', '◆', '６', 
             '‰', 'â', 'ρ', 'φ', '', ""]
    puns = list(set(zh_puncs + en_puncs))
    src_tokenizer = BertTokenizer.from_pretrained(args.dout_bert_dir)   
    tgt_tokenizer = BertTokenizer.from_pretrained(args.din_bert_dir) 
    src_model = BertForMaskedLM.from_pretrained(args.dout_bert_dir)
    tgt_model = BertForMaskedLM.from_pretrained(args.din_bert_dir)
    dict_src, dict_tgt = readdict(args.dictname, puns)
    lines = readfile(args.textname)
    p = Pool(args.pool_num)  
    results = []  
    for i in range(args.pool_num):  
        r = p.apply_async(matched, args=(lines, dict_tgt, dict_src, src_tokenizer, tgt_tokenizer, args.pool_num, i, ))
        results.append(r) 
    p.close() 
    p.join() 
    match_out = [[] for i in range(7)]
    for i in results:
        for j in range(7):
            match_out[j].extend(i.get()[j])
    src_token_texts = match_out[:][0]
    tgt_token_texts = match_out[:][1]
    src_mask_ids = match_out[:][2]
    tgt_mask_ids = match_out[:][3]
    src_word_ids = match_out[:][4]
    tgt_word_ids = match_out[:][5]
    matched_lines = match_out[:][6]
    all_line = len(matched_lines)
    all_batch_idxs = int(all_line) / args.batch_size
    src_input_ids, src_attention_masks = process_input(src_token_texts, src_tokenizer)
    tgt_input_ids, tgt_attention_masks = process_input(tgt_token_texts, tgt_tokenizer)
    src_dataloader = creat_dataloader(src_input_ids, src_attention_masks, args.batch_size)
    tgt_dataloader = creat_dataloader(tgt_input_ids, tgt_attention_masks, args.batch_size)
    src_preds, tgt_preds = predict(src_dataloader, tgt_dataloader, src_model, tgt_model, device,\
        src_mask_ids, tgt_mask_ids, src_word_ids, tgt_word_ids, matched_lines, src_tokenizer, tgt_tokenizer, all_batch_idxs)
    final_lines = transfer(args.max_matched_num)
    writefile(lines, final_lines, args.outname)
    print("over!!")