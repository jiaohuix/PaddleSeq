import sys
import pandas as pd

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.write(''.join(res)) # line
    print(f'write to {file} success.')

def extract_lprob(lines):
    res=[]
    for line in lines:
        if line.startswith("H-"):
            prob=float(line.split("\t")[1])
            res.append(prob)
    return res

def analysis(lprobs):
    df=pd.DataFrame(data={"lprobs":lprobs})
    print(df.describe())

if __name__ == '__main__':
    assert len(sys.argv)==2,f"usage: python {sys.argv[0]} <infile>"
    infile=sys.argv[1]
    lines=read_file(infile)
    lprobs=extract_lprob(lines)
    analysis(lprobs)