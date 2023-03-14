'''
统计长度直方图
'''
import argparse

def read_text_pair(src_file,tgt_file):
    print("Reading text pair...")
    with open(src_file,'r',encoding='utf-8') as fs,open(tgt_file,'r',encoding='utf-8') as ft:
        res = list(zip(fs.readlines(),ft.readlines()))
    return res


def write_text_pair(text_pair_ls,out_src_file,out_tgt_file):
    src_pairs=[pair[0] for pair in text_pair_ls]
    tgt_pairs=[pair[1] for pair in text_pair_ls]
    write_file(src_pairs,out_src_file)
    write_file(tgt_pairs,out_tgt_file)


def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lang id filter")
    parser.add_argument('-t','--hypo',required=True,type=str,default='hypo.txt')
    parser.add_argument('-r','--ref',required=True,type=str,default='ref.txt')
    parser.add_argument('-o','--out',type=str,default="len_hist.png")

    args = parser.parse_args()

    pairs=read_text_pair(args.hypo,args.ref)

    import matplotlib.pyplot as plt
    hypo_len = []
    ref_len = []
    for hypo, ref in pairs:
        hypo, ref = hypo.strip(), ref.strip()
        hypo_len.append(len(hypo.split()))
        ref_len.append(len(ref.split()))
    temp = plt.hist([hypo_len, ref_len], bins=100, rwidth=0.8, histtype="step")
    plt.xlabel("sentence length")
    plt.ylabel("count")
    plt.legend({"hypo_len","ref_len"})
    plt.savefig(args.out)

