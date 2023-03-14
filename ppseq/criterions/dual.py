import paddle
import random
from functools import reduce
from .cross_entropy import CrossEntropyCriterionBase
from ppseq.criterions import register_criterion

@register_criterion(criterion_name="dual_ce")
class CrossEntropyWithDual(CrossEntropyCriterionBase):
    """
    """
    def __init__(self,
                 alpha = 1,
                 label_smooth_eps=0.1,
                 pad_idx=1,
                 max_epochs=50):
        super(CrossEntropyWithDual,self).__init__(label_smooth_eps,pad_idx)
        self.pad_idx =pad_idx
        self.alpha = alpha # backward loss
        self.max_epochs = max_epochs

    def swap_sample(self, sample):
        '''
            input:  [bsz,seq_len]
                    src:   src_text <eos>  1
                    prev:  <eos> tgt_text  2
                    tgt:    tgt_text <eos> 3 [bsz,tgt_len,1]

            output:
                    src:   tgt_text <eos>   (copy 3)
                    prev:  <eos> tgt_text   concat(prev[:,:1],src[:,:-1])
                    tgt:   src_text <eos>   (copy 1)  [bsz,src_len,1]
        '''
        src_tokens = sample["src_tokens"]
        prev_tokens = sample["prev_tokens"]
        reverse_prev_tokens = paddle.concat([prev_tokens[:,:1],src_tokens[:,:-1]],axis=-1)
        reversed_sample = {
                            "src_tokens": sample["tgt_tokens"].squeeze([-1]),
                            "prev_tokens": reverse_prev_tokens,
                            "tgt_tokens":src_tokens.unsqueeze([-1]),
                            }
        return reversed_sample

    def get_pad_mask(self, sample, pad_idx=1):
        # pad 为true
        src_tokens, tgt_tokens = sample["src_tokens"],sample["tgt_tokens"]
        src_len, tgt_len = src_tokens.shape[1], tgt_tokens.shape[1]
        src_pad_mask = (src_tokens == pad_idx).unsqueeze([1]).tile([1, tgt_len, 1])
        tgt_pad_mask = (tgt_tokens == pad_idx).unsqueeze([-1]).tile([1, 1, src_len])
        pad_mask = paddle.where(src_pad_mask,
                                paddle.to_tensor([1.]),
                                paddle.cast(tgt_pad_mask, dtype="float32")
                                )
        return paddle.cast(pad_mask, dtype="bool")

    def attn_loss(self,attn_ft, attn_bt, sample, pad_idx=1):
        # 1. make forward/backward.T attention sore similar (L2 or  -log?)
        attn_ft = attn_ft[:, :-1, 1:] # # [bsz,tgt_len-1,src_len-1]
        attn_bt = attn_bt[:, :-1, 1:] # [bsz,src_len-1,tgt_len-1]
        l2 = (attn_ft - attn_bt.transpose([0,2,1])) ** 2

        # 2. make pad zero
        pad_mask = self.get_pad_mask(sample, pad_idx)[:, :-1, 1:]
        l2 = paddle.where(pad_mask, paddle.to_tensor([0.]), l2)

        token_num = reduce(lambda x, y: x * y, pad_mask.shape) - pad_mask.sum()
        attn_loss = (l2.sum()) / token_num

        # 3. attention more on key location (contrastive)
        # 不仅要让attn 12相近，还要让关键词位置注意力够大（对比学习）

        return attn_loss

    def dual_loss(self):
        # cycle consist
        pass

    def forward(self, model, sample, epoch=None):
        # 1.forward 2.backward 2.ft+bt 3.bt+ft

        '''
        return : loss,sample_size,log
        '''
        # sample={src_tokens, prev_tokens, tgt_tokens}

        # 1. forward ce loss
        logits, sum_cost, avg_cost, token_num, attn_ft = super().forward(model, sample, need_attn=True)
        pred_tokens = paddle.argmax(logits,axis=-1)
        sample["tgt_tokens"] = pred_tokens # take forward model's output into backward model's encoder

        # prob=1 # bidirectional
        prob=0 # bidirectional + dual
        if (epoch is not None) and (self.max_epochs is not None):
            # linear decay, dual prob become largger
            prob = (1/self.max_epochs) * (self.max_epochs - epoch)


        if model.training and random.random()>prob:
            # 2. backward ce loss
            reversed_sample = self.swap_sample(sample)
            logits_bt, sum_cost_bt, avg_cost_bt, token_num_bt, attn_bt = super().forward(model, reversed_sample, need_attn=True)

            # 3.add loss
            sum_cost  +=  self.alpha * sum_cost_bt
            avg_cost += self.alpha * avg_cost
            token_num += token_num_bt

            # attention loss
            beta = 1
            avg_cost += beta * self.attn_loss(attn_ft,attn_bt,sample,self.pad_idx)


        return logits, sum_cost, avg_cost, token_num

