# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
Distillation loss 8/28 0:12
ref: https://github.com/BR-IDL/PaddleViT/blob/develop/image_classification/DeiT/losses.py
'''
import paddle
import paddle.nn.functional as F
from .cross_entropy import CrossEntropyCriterionBase
from ppseq.criterions import register_criterion

@register_criterion(criterion_name="kd_ce")
class DistillationCriterion(CrossEntropyCriterionBase):
    def __init__(self,
                 # kd
                 teacher_model,
                 alpha=0.5,
                 tau=1,
                 label_smooth_eps=0.1,
                 pad_idx=1,
                 ):
        super(DistillationCriterion,self).__init__(label_smooth_eps,pad_idx)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.tau = tau
        self.s2t_vocab_map= None # student2teacher

    def forward(self, model, sample):
        # student output
        logits_s,sum_cost, ce_cost, token_num = super().forward(model, sample)

        # teacher output
        with paddle.no_grad():
            logits_t, _ = self.teacher_model(sample["src_tokens"], sample["prev_tokens"])

        # student vocab map to teacher vocab
        if self.s2t_vocab_map is None:
            self.s2t_vocab_map = []
            for idx_s in range(len(model.tgt_vocab)):
                token_s = model.tgt_vocab.to_tokens(idx_s)
                idx_t = self.teacher_model.tgt_vocab.to_indices(token_s)
                self.s2t_vocab_map.append(idx_t)

        # select correspond teacher vocabd
        logits_t = logits_t.index_select(index=paddle.to_tensor(self.s2t_vocab_map),axis=-1) # [bsz,seq,teacher_vocab] -> [bsz,seq,student_vocab]


        # kd loss
        distillation_loss = F.kl_div(
                F.log_softmax(logits_s/self.tau,axis=-1),
                F.log_softmax(logits_t/self.tau,axis=-1),
                reduction="none") * (self.tau*self.tau)

        # mask pad token's kd loss to 0
        token_mask = paddle.cast(sample["tgt_tokens"]!=self.pad_idx,dtype="float32") # [bsz,seq_len,1]
        distillation_loss = (distillation_loss * token_mask).sum() / token_num

        # final loss
        loss = (1-self.alpha) * ce_cost + self.alpha * distillation_loss

        return logits_s, sum_cost, loss, token_num
