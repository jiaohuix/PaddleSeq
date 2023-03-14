import math
import paddle
import paddle.nn.functional as F

class NMTMetric(paddle.metric.Metric):
    def __init__(self, name='convs2s'):
        self.smooth_loss = 0
        self.nll_loss = 0
        self.steps = 0
        self.gnorm = 0
        self._name = name

    @paddle.no_grad()
    def update(self, sum_loss, logits, target, sample_size, pad_id, gnorm):
        '''
        :return: current batch loss,nll_loss,ppl
        '''
        loss = sum_loss / sample_size / math.log(2)
        nll_loss, ppl = calc_ppl(logits, target, sample_size, pad_id)
        self.smooth_loss += float(loss)
        self.nll_loss += float(nll_loss)
        self.steps += 1
        self.gnorm += gnorm
        return loss, nll_loss, ppl

    def accumulate(self):
        '''
        :return:accumulate batches loss,nll_loss,ppl
        '''
        avg_loss = self.smooth_loss / self.steps
        avg_nll_loss = self.nll_loss / self.steps
        ppl = pow(2, min(avg_nll_loss, 100.))
        gnorm = self.gnorm / self.steps
        return avg_loss, avg_nll_loss, ppl, gnorm

    def reset(self):
        self.smooth_loss = 0
        self.nll_loss = 0
        self.steps = 0
        self.gnorm = 0

    def name(self):
        """
        Returns metric name
        """
        return self._name


@paddle.no_grad()
def calc_ppl(logits, tgt_tokens, token_num, pad_id, base=2):
    tgt_tokens = tgt_tokens.astype('int64')
    nll = F.cross_entropy(logits, tgt_tokens, reduction='sum', ignore_index=pad_id)  # bsz seq_len 1
    nll_loss = nll / token_num / math.log(2)  # hard ce
    nll_loss = min(nll_loss.item(), 100.)
    ppl = pow(base, nll_loss)
    return nll_loss, ppl

