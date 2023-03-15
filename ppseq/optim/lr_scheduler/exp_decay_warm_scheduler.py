import paddle
from .plateau_with_anneal_schedule import ReduceOnPlateauWithAnnael
from ppseq.optim import register_lr_scheduler

@register_lr_scheduler("exp_decay")
def ExpDecayWithWarmup(warmup_steps,min_lr, learning_rate, factor):
    lr_peak = learning_rate
    ''' warmup and exponential decay'''
    # exp_sched = paddle.optimizer.lr.ExponentialDecay(learning_rate=lr_peak, gamma=lr_decay)
    exp_sched = ReduceOnPlateauWithAnnael(learning_rate=lr_peak, factor=factor)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=exp_sched, warmup_steps=warmup_steps,
                                                 start_lr=min_lr, end_lr=lr_peak, verbose=True)
    return scheduler