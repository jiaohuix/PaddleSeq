'''
长度-bleu柱形图
'''
import bisect
import argparse
import numpy as np
from paddlenlp.metrics import BLEU
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def draw_BLEU_Length(y_ls=[],labels=[], savepath="BLEU_Length.png"):
    # eg: draw_BLEU_Length(y_ls=[y1,y2], labels=["sent2sent","doc2doc"])
    num_sys,cols = len(labels), len(y_ls[0])
    bar_width = 0.2
    xticks = ["<25","25-50","50-75","75-100","100-150","150-200","200-250",">250"]
    x = np.array(list(range(cols)))
    idx = 0
    plt.title("BLEU with Length of Sequences")
    for bleu,label in zip(y_ls,labels):
        plt.bar(x+ idx*bar_width, bleu, bar_width, align="center", label=label, alpha=0.5)
        idx += 1
    plt.xlabel("Length of Sequences")
    plt.ylabel("BLEU")
    plt.legend()
    plt.xticks(x+bar_width/num_sys,xticks[:cols]) # tick放到中间
    # plt.show()
    plt.savefig(savepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lang id filter")
    parser.add_argument('-o','--out',type=str,default="belu_len.png")
    parser.add_argument('-t','--tags',required=True,type=str,default='a,b,c')
    parser.add_argument('-r','--ref',required=True,type=str,default='ref.txt')
    parser.add_argument("hypos", nargs="*", help="hypo files")
    args = parser.parse_args()
    system_num = len(args.hypos)
    tags = (args.tags).split(",")
    assert len(tags) == system_num, "tags num != hypos num"
    # read file
    ref_lines = read_file(args.ref)
    hypo_lines_ls = [read_file(file) for file in args.hypos]
    for hypos in hypo_lines_ls:
        assert len(ref_lines) == len(hypos)

    bins = [25, 50, 75, 100, 150, 200, 250]
    # xticks = ["<25","25-50","50-75","75-100","100-150","150-200","200-250",">250"]
    scorer_ls = [[BLEU() for _ in range(len(bins)+1)]
                            for _ in range(system_num)] # [system_idx][bin_idx]
    # 初始化8个scores，迭代每行，对每个系统的hypo记录bleu；结束后去除bleu分，去掉0
    bins_num = [0 for i in range(len(bins)+1)]
    for line_idx in tqdm(range(len(ref_lines))):
        for sys_idx,hypo_lines in enumerate(hypo_lines_ls):
            ref, hypo = ref_lines[line_idx].strip(), hypo_lines[line_idx].strip()
            bin_idx = bisect.bisect_left(bins, len(ref.split()))
            scorer_ls[sys_idx][bin_idx].add_inst(cand=hypo.replace("@@ ","").split(),
                                                 ref_list=[ref.replace("@@ ","").split()])
            bins_num[bin_idx] += 1

    # 记录每个系统的bleu
    zero_bin_idx = bins_num.index(0)

    y_ls = []
    for scorers in scorer_ls: # each system
        bleu_ls = []
        for bin_idx ,scorer in enumerate(scorers): # each bin
            if bin_idx == zero_bin_idx: break
            bleu = scorer.score()
            bleu_ls.append(round(bleu * 100, 3))
        y_ls.append(bleu_ls)

    draw_BLEU_Length(y_ls = y_ls, labels= tags, savepath=args.out)