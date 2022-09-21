import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))
import math
import paddle
from tqdm import tqdm
import paddle.nn.functional as F
import paddle.distributed as dist
from paddlenlp.metrics import BLEU
from paddleseq.reader import prep_vocab
from paddleseq.models import SequenceGenerator
from paddleseq.utils import to_string

import sacrebleu

@paddle.no_grad()
def validation(conf, dataloader, model, criterion, logger):
    # Validation
    model.eval()
    if hasattr(model,"stream"): # dont's support stream evaluation on dev set.
        model.stream=False
    total_smooth = []
    total_nll = []
    total_ppl = []
    dev_bleu = 0
    # for eval bleu
    scorer = BLEU()
    report_bleu = conf.train.report_bleu
    tgt_vocab = prep_vocab(conf)[1] if report_bleu else None
    ignore_symbols = [conf.model.bos_idx, conf.model.pad_idx, conf.model.eos_idx, conf.model.unk_idx]
    hypos_ls = []
    refs_ls = []
    with paddle.no_grad():
        for input_data in tqdm(dataloader):
            # 1.forward loss
            (samples_id, src_tokens, prev_tokens, tgt_tokens) = input_data
            sample = {"src_tokens":src_tokens,"prev_tokens":prev_tokens,"tgt_tokens":tgt_tokens}
            logits, sum_smooth_cost, avg_cost, token_num = criterion(model, sample)
            # logits = model(src_tokens, prev_tokens)[0]
            # sum_smooth_cost, avg_cost, token_num = criterion(logits, tgt_tokens)

            sum_nll_loss = F.cross_entropy(logits, tgt_tokens, reduction="sum", ignore_index=conf.model.pad_idx)
            # 2.gather metric from all replicas
            if dist.get_world_size() > 1:
                dist.all_reduce(sum_smooth_cost)
                dist.all_reduce(sum_nll_loss)
                dist.all_reduce(token_num)

            # 3.caculate avg loss and ppl
            avg_smooth_loss = float(sum_smooth_cost / token_num) / math.log(2)
            avg_nll_loss = float(sum_nll_loss / token_num) / math.log(2)
            avg_ppl = pow(2, min(avg_nll_loss, 100))

            total_smooth.append(avg_smooth_loss)
            total_nll.append(avg_nll_loss)
            total_ppl.append(avg_ppl)

            # 4.record instance for bleu
            if report_bleu:
                if not conf.train.eval_beam:
                    pred_tokens = paddle.argmax(logits, axis=-1) # 添加beam-search [bsz,seq]
                else:
                    bsz = src_tokens.shape[0]
                    samples = {'id': samples_id, 'nsentences': bsz,
                               'net_input': {'src_tokens': paddle.cast(src_tokens, dtype='int64')},
                               'target': tgt_tokens}
                    generator = SequenceGenerator(model=model if not isinstance(model,paddle.DataParallel) else model._layers, vocab_size=conf.model.tgt_vocab_size, beam_size=conf.generate.beam_size,
                                              search_strategy=conf.generate.search_strategy)  # 可以加些参数
                    hypos = generator.generate(samples)
                    pred_tokens = [hypo[0]["tokens"] for hypo in hypos]
                for hypo_tokens, tgt_tokens in zip(pred_tokens, tgt_tokens):
                    hypo_str = to_string(hypo_tokens, tgt_vocab, bpe_symbol="subword_nmt",
                                               extra_symbols_to_ignore=ignore_symbols)
                    tgt_str = to_string(tgt_tokens, tgt_vocab, bpe_symbol="subword_nmt",
                                              extra_symbols_to_ignore=ignore_symbols)
                    # hypos_ls.append(hypo_str)
                    # refs_ls.append(tgt_str)
                    scorer.add_inst(cand=hypo_str.split(), ref_list=[tgt_str.split()])

        avg_smooth_loss = sum(total_smooth) / len(total_smooth)
        avg_nll_loss = sum(total_nll) / len(total_nll)
        avg_ppl = sum(total_ppl) / len(total_ppl)
        bleu_msg = ''
        if report_bleu:
            # sacre_info = str(sacrebleu.corpus_bleu(hypos_ls,[refs_ls]))
            # dev_bleu = float(sacre_info.split()[2])
            try:
                dev_bleu = round(scorer.score() * 100,3)
            except:
                dev_bleu = 0.
            bleu_msg = f"Eval | BLEU Score: {dev_bleu:.3f}"



        logger.info(f"Eval | Avg loss: {avg_smooth_loss:.3f} | nll_loss:{avg_nll_loss:.3f} | ppl: {avg_ppl:.3f} | {bleu_msg}")

    model.train()

    return avg_smooth_loss, avg_nll_loss, avg_ppl, dev_bleu
