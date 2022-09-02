from paddle.optimizer.lr import LRScheduler
class InverseSquareRoot(LRScheduler):
    def __init__(self, warmup_init_lr, warmup_steps, learning_rate=0.1, last_epoch=-1, verbose=False):
        '''
            For Transformer.
        '''
        self.learning_rate = learning_rate
        assert self.learning_rate < 1, "learning_rate must greater than 1 when use inverse_sqrt."
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = warmup_steps

        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (learning_rate - warmup_init_lr) / warmup_steps

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = learning_rate * warmup_steps ** 0.5
        super(InverseSquareRoot, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            return self.decay_factor * self.last_epoch ** -0.5
