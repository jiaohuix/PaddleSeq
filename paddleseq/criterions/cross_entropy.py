from paddlenlp.transformers import CrossEntropyCriterion

class CrossEntropyCriterionBase(CrossEntropyCriterion):
    def __init__(self,
                 label_smooth_eps=None,
                 pad_idx=1 ):
        super(CrossEntropyCriterionBase,self).__init__(label_smooth_eps,pad_idx)

    def forward(self, model, sample ,need_attn=False):
        logits, attn = model(sample["src_tokens"], sample["prev_tokens"])
        sum_cost, avg_cost, token_num = super().forward(logits, sample["tgt_tokens"])
        if not need_attn:
            return logits, sum_cost, avg_cost, token_num
        else:
            return logits, sum_cost, avg_cost, token_num, attn