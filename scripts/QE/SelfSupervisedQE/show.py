def read_file(file_name):
    with open(file_name, "r", encoding="UTF-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

if __name__ == "__main__":
    src_all = read_file("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/SelfSupervisedQE/data/sample.tok.zh")
    pred_all = read_file("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/SelfSupervisedQE/data/sample.tok.en")
    ref_all = read_file("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/SelfSupervisedQE/data/sample.tok.ref")
    score_all = read_file("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/SelfSupervisedQE/logs/score.out")
    align_all = read_file("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/SelfSupervisedQE/data/sample.align")
    
    with open("/data/zhanghongxiao/NMT/NN_Domain_mg/gjl/SelfSupervisedQE/logs/sample.show", "w", encoding="UTF-8") as f:

        for src, pred, ref, score, align in zip(src_all, pred_all, ref_all, score_all, align_all):
            f.write("src:\t")
            for i, s in enumerate(src.split()):
                f.write(s + '@{} '.format(i))
            f.write('\n')
            f.write("pred:\t")
            for i, s in enumerate(pred.split()):
                f.write(s + '@{} '.format(i))
            f.write('\n')
            f.write("ref:\t{}\n".format(ref))
            f.write("score:\t")
            for i, s in enumerate(score.split()):
                if(float(s) > 3): f.write("BAD")
                else: f.write("OK")
                f.write("@{} ".format(i))
                # f.write("{:.2f}@{} ".format(float(s), i))
            f.write("\n")
            f.write("align:\t{}\n".format(align))
            f.write("\n")