from paddlenlp.losses import RDropLoss
from .cross_entropy import CrossEntropyCriterionBase
from ppseq.criterions import register_criterion

@register_criterion(criterion_name="rdrop_ce")
class CrossEntropyWithRdrop(CrossEntropyCriterionBase):
    """
    """
    def __init__(self,
                 label_smooth_eps=0.1,
                 pad_idx=1,
                 alpha=5):
        super(CrossEntropyWithRdrop,self).__init__(label_smooth_eps,pad_idx)
        self.pad_idx =pad_idx
        self.alpha= alpha
        self.rdrop_loss = RDropLoss()

    def forward(self, model, sample, need_attn=False):
        '''
        return : loss,sample_size,log
        '''
        # 1.loss ce
        logits, sum_cost, avg_cost, token_num = super().forward(model, sample, need_attn=False)

        # 2.rdrop loss
        if model.training:
            avg_cost = self.get_rdrop_loss(model, sample, logits, avg_cost)

        return logits, sum_cost, avg_cost, token_num

    def get_rdrop_loss(self, model, sample, logits1 ,ce_loss1):
        if self.alpha > 0:
            logits2, sum_cost2, ce_loss2, token_num2 = super().forward(model, sample)
            pad_mask = (sample["prev_tokens"] != self.pad_idx).unsqueeze(-1).tile([1,1,logits1.shape[-1]])
            kl_loss = self.rdrop_loss(logits1,logits2,pad_mask)
            kl_loss = kl_loss/token_num2
            loss = 0.5 * (ce_loss1 + ce_loss2) + self.alpha * kl_loss
        else:
            loss = ce_loss1

        return loss