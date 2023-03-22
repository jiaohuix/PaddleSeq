# task用于获取数据和模型等组件；以及最小的动作单元，如训练、推理一个step；
# 而train和trainer负责外部的训练循环，包括epoch循环、模型加载保存、早停、进度日志，都是些通用策略。

class BaseTask(object):
    def __init__(self):
        pass

    @classmethod
    def setup_task(cls):
        return cls

    def load_dataset(self):
        pass

    def load_model(self):
        pass

    def load_criterion(self):
        pass

    def load_optimizer(self):
        pass

    def train_step(self):
        pass

    def valid_step(self):
        pass

    def inference(self):
        pass
