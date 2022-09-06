import os
import paddle
from .transformer import *
from .seq_generator import SequenceGenerator
import paddleseq.models as models
from paddleseq.reader import prep_vocab
from paddleseq.checkpoint_utils import freeze_by_names,unfreeze_by_names
from yacs.config import CfgNode


def build_model(conf_or_path,is_test=False):
    if isinstance(conf_or_path,CfgNode):
        conf = conf_or_path
    elif isinstance(conf_or_path,str) and os.path.isfile(os.path.join(conf_or_path,'model.yaml')): # load config
        args_path = os.path.join(conf_or_path,'model.yaml')
        assert os.path.isfile(args_path), "conf path should not be empty!"
        conf = CfgNode.load_cfg(open(args_path, encoding="utf-8"))
    else:
        raise ValueError("conf_or_path is is neither CfgNode nor pretrained path error.")

    model_args,gen_args=conf.model,conf.generate
    src_vocab, tgt_vocab = prep_vocab(conf)

    model_path=os.path.join(model_args.init_from_params,'model.pdparams')
    model_path=None if not os.path.exists(model_path) else model_path
    model=getattr(models,model_args.model_name)(
                                        is_test=is_test,
                                        pretrained_path=model_path,
                                        src_vocab = src_vocab,
                                        tgt_vocab = tgt_vocab,
                                        max_length=model_args.max_length,
                                        dropout=model_args.dropout)

    # 1.about embed dict
    # if model_path is not None:
    #     # save src vocab
    #     model.save_embedding(model_args.init_from_params, model.src_vocab, model.src_word_embedding) # path/vocab.npy
    # if not model.share_embed:
    #     print("no share, load share vocab to tgt_embed!")
    #     model.load_embedding(model_args.init_from_params,model.tgt_vocab, model.tgt_word_embedding)

    # 2.freeze layers exclude output_projection
    # freeze_by_names(model,exclude_layers=["output_projection"])

    return model