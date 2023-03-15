import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import CrossEntropyCriterion
from ppseq.criterions import register_criterion

@register_criterion(criterion_name="simcut_ce")
class CrossEntropyWithSimcut(nn.Layer):
    """
    loss= CE( P(y'|x),y ) + alpha * KL(P(y'|x)||P(y'|x_cut))
    alpha: alpha for simcut, 0 means no simcut"
    simcut_p:  probability for cutoff in simcut, 0 means no cutoff in simcut
    """
    def __init__(self,
                 label_smooth_eps=0.1,
                 pad_idx=1,
                 alpha=0.0,
                 simcut_p=0.0):
        super(CrossEntropyWithSimcut,self).__init__()
        self.ce_criterion=CrossEntropyCriterion(label_smooth_eps=label_smooth_eps,pad_idx=pad_idx)
        self.pad_idx =pad_idx
        self.alpha= alpha
        self.simcut_p=simcut_p

    def forward(self, model, sample):
        '''
        return : loss,sample_size,log
        '''
        # sample={src_tokens, prev_tokens, tgt_tokens}

        # 1.loss ce
        logits_orig, attn = model(sample["src_tokens"], sample["prev_tokens"])
        sum_cost, avg_cost, token_num = self.ce_criterion(logits_orig, sample["tgt_tokens"])

        # 2. loss kl
        if model.training:
            logits_cut, sum_cost_kl, avg_cost_kl, token_num = self.simcut(model,logits_orig,sample)
            # 两份交叉熵
            # sum_cost_2, avg_cost_2, token_num = self.ce_criterion(logits_cut, sample["tgt_tokens"])
            # sum_cost = 0.5 * (sum_cost + sum_cost_2)
            # avg_cost = 0.5 * (avg_cost + avg_cost_2)

            sum_cost += self.alpha * sum_cost_kl
            avg_cost += self.alpha * avg_cost_kl

        return logits_orig, sum_cost, avg_cost, token_num

    def kl_div(self,p,q):
        kl = F.kl_div(input=F.log_softmax(q), label=F.softmax(p), reduction="none")  # [bsz,seq,vocab]
        return kl

    def simcut(self,model,logits_orig,sample):
        valid_indices = paddle.cast(sample["tgt_tokens"] != self.pad_idx,dtype=logits_orig.dtype) # tgt_tokens [bsz,seq,1]

        # 1.logits
        encoder_out = model.forward_encoder(sample["src_tokens"],simcut_p=self.simcut_p)
        logits_cut, attn = model.forward_decoder(sample["prev_tokens"], encoder_out,simcut_p=self.simcut_p)

        # 2.pq
        cost = self.kl_div(p=logits_orig,q=logits_cut)
        # cost = 0.5 * self.kl_div(p=logits_orig,q=logits_cut) + 0.5 * self.kl_div(p=logits_cut,q=logits_orig)

        # p = F.softmax(logits_orig)
        # q = F.log_softmax(logits_cut)
        # # 3.kl
        # cost = F.kl_div(input=q,label=p,reduction="none") # [bsz,seq,vocab]

        # 4.mask pad
        sum_cost = (cost * valid_indices).sum() # [1]
        token_num = paddle.sum(valid_indices)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num

        return logits_cut,sum_cost, avg_cost, token_num