import paddle
from .plateau_with_anneal_schedule import ReduceOnPlateauWithAnnael
def ExpDecayWithWarmup(warmup_steps, lr_start, lr_peak, lr_decay):
    ''' warmup and exponential decay'''
    # exp_sched = paddle.optimizer.lr.ExponentialDecay(learning_rate=lr_peak, gamma=lr_decay)
    exp_sched = ReduceOnPlateauWithAnnael(learning_rate=lr_peak, factor=lr_decay)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=exp_sched, warmup_steps=warmup_steps,
                                                 start_lr=lr_start, end_lr=lr_peak, verbose=True)
    return scheduler