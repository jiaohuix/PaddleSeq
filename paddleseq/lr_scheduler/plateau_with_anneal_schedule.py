import numpy as np
from paddle import Tensor
from paddle.optimizer.lr import ReduceOnPlateau

class ReduceOnPlateauWithAnnael(ReduceOnPlateau):
    '''
        For ConvS2S.
        Reduce learning rate when ``metrics`` has stopped descending. Models often benefit from reducing the learning rate
    by 2 to 10 times once model performance has no longer improvement.
        [When lr is not updated for force_anneal times,then force shrink the lr by factor.]
    '''

    def __init__(self,
                 learning_rate,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 force_anneal=50,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 epsilon=1e-8,
                 verbose=False,
                 ):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.force_anneal = args.pop('force_anneal')
        super(ReduceOnPlateauWithAnnael, self).__init__(**args)
        self.num_not_updates = 0

    def state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr', 'num_not_updates'
        ]

    def step(self, metrics, epoch=None):
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch

        # loss must be float, numpy.ndarray or 1-D Tensor with shape [1]
        if isinstance(metrics, (Tensor, np.ndarray)):
            assert len(metrics.shape) == 1 and metrics.shape[0] == 1, "the metrics.shape " \
                                                                      "should be (1L,), but the current metrics.shape is {}. Maybe that " \
                                                                      "you should call paddle.mean to process it first.".format(
                metrics.shape)
        elif not isinstance(metrics,
                            (int, float, np.float32, np.float64)):
            raise TypeError(
                "metrics must be 'int', 'float', 'np.float', 'numpy.ndarray' or 'paddle.Tensor', but receive {}".
                    format(type(metrics)))

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best is None or self._is_better(metrics, self.best):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:  # >=patience，update lr，and set annel=0
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                self.num_not_updates = 0
                new_lr = max(self.last_lr * self.factor, self.min_lr)
                if self.last_lr - new_lr > self.epsilon:
                    self.last_lr = new_lr
                    if self.verbose:
                        print('Epoch {}: {} set learning rate to {}.'.format(
                            self.last_epoch, self.__class__.__name__,
                            self.last_lr))
            else:  # Update here
                self.num_not_updates += 1
                if self.num_not_updates >= self.force_anneal:
                    self.num_not_updates = 0
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
                    new_lr = max(self.last_lr * self.factor, self.min_lr)
                    if self.last_lr - new_lr > self.epsilon:
                        self.last_lr = new_lr
                        if self.verbose:
                            print('Epoch {}: {} set learning rate to {} because of force anneal.'.format(
                                self.last_epoch, self.__class__.__name__,
                                self.last_lr))


def force_anneal(scheduler: ReduceOnPlateau, anneal: int):
    setattr(scheduler, 'force_anneal', anneal)
    setattr(scheduler, 'num_not_updates', 0)

    def state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr', 'num_not_updates'
        ]

    setattr(scheduler, 'state_keys', state_keys)

    def step(self, metrics, epoch=None):
        pass

    setattr(scheduler, 'step', step)
    return scheduler
