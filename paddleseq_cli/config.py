import paddle
import argparse
from yacs.config import CfgNode


def get_arguments(return_parser=False):
    """return argumeents, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser(description="Simultranslation", add_help=True)
    parser.add_argument("-c", "--cfg", default=None, type=str,required=True, metavar="FILE", help="yaml file path")

    # distributed training parameters
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ngpus", default=-1, type=int)
    parser.add_argument("--update-freq", default=None, type=int)

    parser.add_argument("--max-epoch", default=None, type=int)
    parser.add_argument("--save-epoch", default=None, type=int)
    parser.add_argument("--save-dir", default=None, type=str,help="save dir for model、log、generated text")
    parser.add_argument("--resume", default="", type=str, help="resume from checkpoint")
    parser.add_argument("--last-epoch", default=None, type=int, help="resume from epoch+1")
    parser.add_argument("--last-step", default=None, type=int, help="resume from step+1")
    parser.add_argument("--log-steps", default=None, type=int, help="Number of steps between log print.")
    parser.add_argument("--report-bleu", action="store_true",help="report bleu when valid")
    parser.add_argument("--eval-beam", action="store_true",help="use beam search when valid, default argmax.") # 8/19

    # Dataset parameters
    parser.add_argument("--src-lang", default=None, type=str)
    parser.add_argument("--tgt-lang", default=None, type=str)
    parser.add_argument("--only-src", action="store_true")
    parser.add_argument("--train-pref", default=None, type=str)
    parser.add_argument("--valid-pref", default=None, type=str)
    parser.add_argument("--test-pref", default=None, type=str)
    parser.add_argument("--vocab-pref", default=None, type=str)

    parser.add_argument("--max-tokens", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--pad-vocab", action="store_true")

    # Model parameters
    parser.add_argument("--arch", default=None, type=str, help="Name of model to train")
    parser.add_argument("--drop", default=None, type=float, help="Dropout rate")
    parser.add_argument("--pretrained", default=None, type=str, help="pretrained dir")
    parser.add_argument("--save-model", default=None, type=str, help="model save dir")

    # Optimizer parameters
    parser.add_argument("--optim", default=None, type=str, help="Optimizer,support [nag|adam|adamw]")
    parser.add_argument("--clip-norm", default=None, type=float, help="Clip gradient norm")
    parser.add_argument("--momentum", default=None, type=float, help="momentum")
    parser.add_argument("--weight-decay", default=None, type=float, help="weight decay")

    # Learning rate schedule parameters
    parser.add_argument("--lr", default=None, type=float, help="learning rate")
    parser.add_argument("--sched", default=None, type=str, help="LR scheduler, support [plateau|wamup|cosine|noamdecay|inverse_sqrt]")
    parser.add_argument("--warmup", default=None, type=int, help="warmup steps")
    parser.add_argument("--reset-lr",action="store_true",help="weather to reset learning rate to lr when in resume.")
    parser.add_argument("--min-lr", default=None, type=float, help="lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument("--lr-shrink", default=None, type=float, help="lr shrink factor")
    parser.add_argument("--patience", default=None, type=int, help="patience epochs for Plateau LR scheduler")
    parser.add_argument("--force-anneal", default=None, type=int, help="anneal epochs for Plateau LR scheduler")

    # Augmentation parameters
    parser.add_argument("--smoothing", default=None, type=float, help="Label smoothing")

    # Generation parameters
    parser.add_argument("--beam-size", default=None, type=int, help="beam search size")
    parser.add_argument("--infer-bsz", default=None, type=int, help="inference batch size")
    parser.add_argument("--n-best", default=None, type=int)
    parser.add_argument("--generate-path", default=None, type=str)
    parser.add_argument("--sorted-path", default=None, type=str)
    parser.add_argument("--remain-bpe", action="store_true",help="Weather to remain bpe after decode.")
    parser.add_argument("--detokenize", action="store_true",help="Weather to use moses detokenizer.")


    if not return_parser:
        args = parser.parse_args()
        return args
    else:
        return parser

def get_config(args):
    conf = CfgNode.load_cfg(open(args.cfg, encoding="utf-8"))
    conf.defrost()
    # distributed training parameters
    if args.amp:
        conf.train.amp = args.amp
    if args.ngpus:
        conf.ngpus = len(paddle.static.cuda_places()) if args.ngpus == -1 else args.ngpus
    if args.eval:
        conf.eval = args.eval
    conf.eval_beam = args.eval_beam
    if args.update_freq:
        conf.train.update_freq = args.update_freq
    if args.max_epoch:
        conf.train.max_epoch=args.max_epoch
    if args.save_epoch:
        conf.train.save_epoch=args.save_epoch
    if args.save_dir:
        conf.SAVE=args.save_dir
    if args.resume:  # 路径
        conf.train.resume = args.resume
    if args.last_epoch:
        conf.train.last_epoch = args.last_epoch
    if args.last_step:
        conf.train.last_step = args.last_step
    if args.log_steps:
        conf.train.log_steps = args.log_steps
    if args.report_bleu:
        conf.train.report_bleu = args.report_bleu
    # Dataset parameters
    if args.src_lang:
        conf.data.src_lang = args.src_lang
    if args.tgt_lang:
        conf.data.tgt_lang = args.tgt_lang
    if args.only_src:
        conf.data.has_target=False
    if args.train_pref:
        conf.data.train_pref = args.train_pref
    if args.valid_pref:
        conf.data.valid_pref = args.valid_pref
    if args.test_pref:
        conf.data.test_pref = args.test_pref
    if args.vocab_pref:
        conf.data.vocab_pref = args.vocab_pref
    if args.pad_vocab:
        conf.data.pad_vocab = args.pad_vocab
    if args.max_tokens:
        conf.train.max_tokens = args.max_tokens
    if args.seed:
        conf.seed = args.seed
    if args.num_workers:
        conf.train.num_workers = args.num_workers
    # Model parameters
    if args.arch:
        conf.model.model_name = args.arch
    if args.drop:
        conf.model.dropout = args.drop
    if args.pretrained:
        conf.model.init_from_params = args.pretrained
    if args.save_model:
        conf.model.save_model = args.save_model
    # Optimizer parameters
    if args.optim:
        conf.learning_strategy.optim = args.optim
    if args.clip_norm:
        conf.learning_strategy.clip_norm = args.clip_norm
    if args.momentum:
        conf.learning_strategy.optimizer.nag.momentum = args.momentum
    if args.weight_decay:
        conf.learning_strategy.weight_decay = args.weight_decay
    # Learning rate schedule parameters
    if args.sched:
        conf.learning_strategy.sched = args.sched
    if args.lr:
        conf.learning_strategy.learning_rate = args.lr
    if args.warmup:
        conf.learning_strategy.scheduler.warmup.warm_steps = args.warmup
    if args.reset_lr:
        conf.learning_strategy.reset_lr = args.reset_lr
    if args.min_lr:
        conf.learning_strategy.min_lr = args.min_lr
    if args.lr_shrink:
        conf.learning_strategy.scheduler.plateau.lr_shrink = args.lr_shrink
    if args.patience:
        conf.learning_strategy.scheduler.plateau.patience = args.patience
    if args.force_anneal:
        conf.learning_strategy.scheduler.plateau.force_anneal = args.force_anneal
    # Augmentation parameters
    if args.smoothing:
        conf.learning_strategy.label_smooth_eps = args.smoothing
    # Generation parameters
    if args.beam_size:
        conf.generate.beam_size = args.beam_size
    if args.infer_bsz:
        conf.generate.infer_bsz = args.infer_bsz
    if args.n_best:
        conf.generate.n_best = args.n_best
    if args.generate_path:
        conf.generate.generate_path = args.generate_path
    if args.sorted_path:
        conf.generate.sorted_path = args.sorted_path
    if args.remain_bpe:
        conf.generate.remain_bpe = args.remain_bpe
    if args.detokenize:
        conf.generate.detokenize = args.detokenize
    conf.freeze()
    return conf
