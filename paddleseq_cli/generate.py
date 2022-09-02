# -*- coding: utf-8 -*-
import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))
import math
import logging
import paddle
import sacrebleu
from paddlenlp.metrics import BLEU
from sacremoses import MosesDetokenizer,MosesDetruecaser
from paddleseq_cli.config import get_arguments, get_config
from paddleseq.utils import sort_file,post_process,set_paddle_seed,to_string
from paddleseq.reader import prep_dataset,prep_vocab, prep_loader
from paddleseq.models import build_model, SequenceGenerator


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("paddleseq_cli.generate")


@paddle.no_grad()
def generate(conf):
    set_paddle_seed(seed=conf.seed)
    # file op
    if not os.path.exists(conf.SAVE): os.makedirs(conf.SAVE)
    generate_path = os.path.join(conf.SAVE, conf.generate.generate_path)
    sorted_path = os.path.join(conf.SAVE, conf.generate.sorted_path)
    out_file = open(generate_path, 'w', encoding='utf-8') if conf.generate.generate_path else sys.stdout

    logger.info(f'configs:\n{conf}')
    # dataloader
    test_dset = prep_dataset(conf, mode='test')
    test_loader = prep_loader(conf, test_dset, mode='test')
    src_vocab, tgt_vocab = prep_vocab(conf)
    # model
    logger.info('Loading models...')
    model = build_model(conf, is_test=True)
    model.eval()
    # logger.info(f"model architecture:\n{model}")
    scorer = BLEU()
    generator = SequenceGenerator(model, vocab_size=model.tgt_vocab_size, beam_size=conf.generate.beam_size,
                                  search_strategy=conf.generate.search_strategy)

    # 1.for batch
    hypos_ls=[]
    refs_ls=[]
    logger.info('Pred | Predicting...')
    has_target = conf.data.has_target
    for batch_id, batch_data in enumerate(test_loader):
        print(f'batch_id:[{batch_id + 1}/{len(test_loader)}]')
        samples_id, src_tokens, tgt_tokens, tgt_langs = None, None, None, None
        if has_target:
            samples_id, src_tokens, tgt_tokens = batch_data
        else:
            if conf.data.lang_embed:
                samples_id, src_tokens , tgt_lang_tokens = batch_data
            else:
                samples_id, src_tokens = batch_data
        bsz = src_tokens.shape[0]
        samples = {'id': samples_id, 'nsentences': bsz,
                   'net_input': {'src_tokens': paddle.cast(src_tokens, dtype='int64')},
                   'target':tgt_tokens}

        bos_token=None
        hypos = generator.generate(samples,bos_token=bos_token)

        # 2.for sample
        for i, sample_id in enumerate(samples["id"].tolist()):
            # decode src and tgt ,then print
            src_text = post_process(sentence=" ".join(src_vocab.to_tokens(test_dset[sample_id][0])),
                                          symbol='subword_nmt')
            print("S-{}\t{}".format(sample_id, src_text), file=out_file)
            if has_target:
                tgt_text = post_process(sentence=" ".join(tgt_vocab.to_tokens(test_dset[sample_id][1])),
                                              symbol='subword_nmt')
                print("T-{}\t{}".format(sample_id, tgt_text), file=out_file)

            # 3.for prediction
            for j, hypo in enumerate(hypos[i][: conf.generate.n_best]):  # take best n hypo from sample_i's  hypos
                # 3.1 postprocess to hypo
                hypo_str = to_string(hypo["tokens"], tgt_vocab, bpe_symbol='subword_nmt',
                                           extra_symbols_to_ignore=[model.bos_id, model.eos_id, model.pad_id])

                # detokenize
                if conf.generate.detokenize:
                    detok=MosesDetokenizer(lang=conf.data.tgt_lang)
                    hypo_str=detok.detokenize(hypo_str.split())
                    tgt_text=detok.detokenize(tgt_text.split())

                # 3.2 log score info
                score = (hypo["score"] / math.log(2)).item()
                print("H-{}\t{:.4f}\t{}".format(sample_id, score, hypo_str), file=out_file)
                print(
                    "P-{}\t{}".format(sample_id,
                                      " ".join(
                                          map(lambda x: "{:.4f}".format(x),
                                              # convert from base e to base 2
                                              (hypo["positional_scores"] / math.log(2)).tolist(),
                                              )
                                      ),
                                      ),
                    file=out_file
                )
                # 3.3 record score, both sacrebleu and paddle bleu(close to multi-bleu.perl)
                if has_target and j == 0:
                    scorer.add_inst(cand=hypo_str.split(), ref_list=[tgt_text.split()])
                    # sacrebleu
                    hypos_ls.append(hypo_str)
                    refs_ls.append(tgt_text)

    # report final bleu score
    if has_target:
        logger.info(f"Paddle BlEU Score:{scorer.score() * 100:.4f}")
        print(F"Sacrebleu: {sacrebleu.corpus_bleu(hypos_ls, [refs_ls])}")

    if conf.generate.generate_path and conf.generate.sorted_path:
        sort_file(gen_path=generate_path, out_path=sorted_path)

if __name__ == '__main__':
    args = get_arguments()
    conf = get_config(args)
    generate(conf)
