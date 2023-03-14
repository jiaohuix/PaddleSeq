from paddlenlp.transformers import CrossEntropyCriterion
from ppseq.criterions import register_criterion

@register_criterion(criterion_name="ce")
class CrossEntropyCriterionBase(CrossEntropyCriterion):
    def __init__(self,
                 label_smooth_eps=0.1,
                 pad_idx=1):
        super(CrossEntropyCriterionBase,self).__init__(label_smooth_eps,pad_idx)

    def forward(self, model, sample ,need_attn=False):
        logits, attn = model(sample["src_tokens"], sample["prev_tokens"])
        sum_cost, avg_cost, token_num = super().forward(logits, sample["tgt_tokens"])
        if not need_attn:
            return logits, sum_cost, avg_cost, token_num
        else:
            return logits, sum_cost, avg_cost, token_num, attn