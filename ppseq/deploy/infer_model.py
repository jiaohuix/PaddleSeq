'''
Infermodel将model和generator组合起来，并处理输入输出，在export中加载配置
'''


import paddle
import paddle.nn as nn


# infer只实现输入和输出tokens，  predictor来实现输入和输出处理
# 包装src_tokns
class NMTInferModel(nn.Layer):
    def __init__(self,model,generator):
        super(NMTInferModel,self).__init__()
        self.model = model
        self.generator = generator

    def eval(self):
        self.model.eval()

    def prep_samples(self,src_tokens): # [bsz,seq]
        bsz = src_tokens.shape[0]
        samples_id = paddle.arange(0, bsz)
        samples = {'id': samples_id, 'nsentences': bsz,
                   'net_input': {'src_tokens': src_tokens.astype("int64")},
                   'target': None}
        return samples

    def forward(self, src_tokens):
        samples = self.prep_samples(src_tokens)
        hypos = self.generator.generate(samples)  #[bsz* [beam* item]],每个样本有beam个list, item={'tokens', 'score', 'attention', 'alignment', 'positional_scores'}
        output_tokens = [hypo[0]["tokens"] for hypo in hypos]
        return output_tokens

    def beam_search(self):
        pass








