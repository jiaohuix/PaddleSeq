import os
import random
import paddle
import numpy as np

def set_paddle_seed(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    except:
        pass


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def strip_pad(tensor, pad_id):
    return tensor[tensor != pad_id]


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re
        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(' +', ' ', sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


def to_string(
        tokens,
        vocab,
        bpe_symbol=None,
        extra_symbols_to_ignore=None,
        separator=" "):
    extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
    tokens = [int(token) for token in tokens if int(token) not in extra_symbols_to_ignore]  # 去掉extra tokens
    sent = separator.join(
        vocab.to_tokens(tokens)
    )
    return post_process(sent, bpe_symbol)


def sort_file(gen_path="generate.txt", out_path="result.txt"):
    result = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("H-"):
                result.append(line.strip())
    result = sorted(result, key=lambda line: int(line.split("\t")[0].split("-")[1]))
    result = [line.split("\t")[2].strip().replace("\n","")+"\n" for line in result] # 单句出现\n会导致行数不一致！
    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write("".join(result))
    print(f"write to file {out_path} success.")


def get_grad_norm(grads):
    norms = paddle.stack([paddle.norm(g, p=2) for g in grads if g is not None])
    gnorm = paddle.norm(norms, p=2)
    return float(gnorm)










