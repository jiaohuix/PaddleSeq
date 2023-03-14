from paddle.optimizer.lr import LRScheduler
class KneeLRScheduler(LRScheduler):

    def __init__(self,warmup_init_lr , peak_lr, warmup_steps=0, explore_steps=0, total_steps=0, last_epoch=-1, verbose=False):
        '''
        paper: Wide-minima Density Hypothesis and the Explore-Exploit Learning Rate Schedule
        last_epoch use as steps
        '''
        self.warmup_init_lr = warmup_init_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.current_step = 1

        assert self.decay_steps >= 0

        super(KneeLRScheduler, self).__init__(peak_lr, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps and self.warmup_steps>0:# avoid -1<0
            return self.peak_lr * self.last_epoch / self.warmup_steps
        elif self.last_epoch  <= (self.explore_steps + self.warmup_steps):
            return self.peak_lr
        else:
            slope = -1 * self.peak_lr / self.decay_steps
            return max(0.0, self.peak_lr + slope * (self.last_epoch - (self.explore_steps + self.warmup_steps)))
