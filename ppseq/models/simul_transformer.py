# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
modify:
1.remove static_cache
2.add target key pad mask
3.add forward_encoder,forward_decoder,reorder_encoder_out,reorder_incremental_state, in order to adapt to fairseq generator
5.fairseq attn_drop=act_drop=0,paddle default attn_drop=act_drop=drop

2023/3/1
未初始化：transformer的linear1、linear2，attention的qkv bias

待优化：
1.加入prefix或context
2.训练时的调度采样
'''
from __future__ import print_function
import math
import types
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import PositionalEmbedding
import paddle.nn.initializer as I
from fastcore.all import patch_to, partial
from ppseq.modules.initializer import xavier_uniform_

@patch_to(nn.Layer)
def apply(self, fn, name=""):
    for n, layer in self.named_children():
        nnmame = n if name == "" else name + "." + n
        layer.apply(fn, nnmame)

    fn(self, name)
    return self

class DecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(DecoderLayer, self).__init__(*args, **kwargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        ''' memory_mask: [bsz,1,1,src_len]'''
        ############### self attention can cache k,v ###############
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        ############### waitk cross attention cannot cache k,v ###############
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if len(memory) == 1:
            # Full sent # autoregressive
            tgt = self.cross_attn(tgt, memory[0], memory[0], memory_mask, None)
        else:
            # Wait-k policy # 训练时根据多个增长的src,预测多个tgt
            cross_attn_outputs = []
            for i in range(tgt.shape[1]):
                q = tgt[:, i:i + 1, :]
                if i >= len(memory):  # 越界取全部src
                    e = memory[-1]
                else:
                    e = memory[i]
                src_len = e.shape[1]
                # cross_attn_outputs.append(self.cross_attn(q, e, e, memory_mask[:, :, i:i+1, :src_len], None)) # tgt取1个,src取前几个 e[bsz,len,dim]
                cross_attn_outputs.append(
                    self.cross_attn(q, e, e, memory_mask[:, :, :, :src_len], None))  # tgt取1个,src取前几个 e[bsz,len,dim]
            tgt = paddle.concat(cross_attn_outputs, axis=1)
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,))

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return (incremental_cache,)


class Decoder(nn.TransformerDecoder):
    """
    PaddlePaddle 2.1 casts memory_mask.dtype to memory.dtype, but in STACL,
    type of memory is list, having no dtype attribute.
    memory: list
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_caches(self, memory):
        ''' [(increment_cache,),...] ,no static cache for waitk'''
        caches = [layer.gen_cache(memory) for layer in self.layers]
        return caches

    def reorder_cache(self, cache, new_order):
        new_k = paddle.index_select(cache.k, index=new_order, axis=0)
        new_v = paddle.index_select(cache.v, index=new_order, axis=0)
        return type(cache)(new_k, new_v)

    def reorder_incremental_state(self, caches, new_order):
        ''' incremental_state就是caches, 在waitk中是[(Cache,)*layers],只有decoder self attn,无cross attn
            new_order: [bsz*beam_size]
        '''
        new_caches = []
        for cache_tuple in caches:
            increment_cache = self.reorder_cache(cache_tuple[0], new_order)
            new_caches.append(tuple([increment_cache]))
        return new_caches


class SimultaneousTransformer(nn.Layer):
    """
    model
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 waitk=-1,
                 d_model=512,
                 encoder_layers=6,
                 decoder_layers=6,
                 nheads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 weight_sharing=False,
                 max_length=1024,
                 stream=False,
                 bos_id=0,
                 eos_id=2,
                 pad_id=1,
                 unk_id=3):
        super(SimultaneousTransformer, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_dim = d_model
        self.stream = stream  # is test data stream file
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.dropout = dropout
        self.waitk = waitk
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.d_model = d_model
        self.inf = 1e9

        self.src_word_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=d_model,
                                               weight_attr=I.Normal(0, std=d_model ** -0.5))
        self.src_word_embedding.weight.stop_gradient = True
        self.src_word_embedding.weight[self.pad_id, :] = 0.
        self.src_word_embedding.weight.stop_gradient = False
        self.src_pos_embedding = PositionalEmbedding(
            emb_dim=d_model, max_length=max_length + self.pad_id + 1)
        if weight_sharing:
        # if False:
            assert src_vocab_size == tgt_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.tgt_word_embedding = self.src_word_embedding
            self.tgt_pos_embedding = self.src_pos_embedding
        else:
            self.tgt_word_embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=d_model,
                                                   weight_attr=I.Normal(0, std=d_model ** -0.5))
            self.tgt_word_embedding.weight.stop_gradient = True
            self.tgt_word_embedding.weight[self.pad_id, :] = 0.
            self.tgt_word_embedding.weight.stop_gradient = False
            self.tgt_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length + self.pad_id + 1)
        self.embed_scale = d_model ** 0.5
        # encoder_layer = nn.TransformerEncoderLayer(
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=0.,
            act_dropout=0.,
            activation='relu',
            normalize_before=False,
            bias_attr=[True, True])
        encoder_norm = None
        # encoder_norm = nn.LayerNorm(d_model)
        # self.encoder = nn.TransformerEncoder(
        self.encoder = self.modify_encoder(
            nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.encoder_layers, norm=encoder_norm))

        decoder_layer = DecoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=0.,
            act_dropout=0.,
            activation='relu',
            normalize_before=False,
            bias_attr=[True, True, True])
        decoder_norm = None
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = self.modify_decoder(Decoder(decoder_layer=decoder_layer, num_layers=n_layer, norm=decoder_norm))
        self.decoder = Decoder(decoder_layer=decoder_layer, num_layers=self.decoder_layers, norm=decoder_norm)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(
                x=x, y=self.tgt_word_embedding.weight, transpose_y=True)
        else:
            self.linear = nn.Linear(
                in_features=d_model,
                out_features=tgt_vocab_size,
                bias_attr=False,
                weight_attr=I.Normal(0, std=d_model ** -0.5))
        self.apply(self.reset_paramaters)

    def reset_paramaters(self,m,n): # model,name
        if isinstance(m,nn.Linear):
            in_features = m.weight.shape[0]
            uniform_ = I.Uniform(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features))
            if any(x in n for x in ["q_proj", "k_proj", "v_proj"]):
                xavier_uniform_(m.weight,gain=2**-0.5)
                if m.bias is not None:
                    uniform_(m.bias)
            elif any(x in n for x in ["out_proj"]):
                xavier_uniform_(m.weight)
            elif any(x in n for x in ["linear1","linear2"]):
                uniform_(m.weight)
                if m.bias is not None:
                    uniform_(m.bias)

        # weight_attr = init.Uniform(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features)),
            # bias_attr = init.Uniform(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features)) if bias else bias)
    def modify_encoder(self, encoder):
        ''' 为encoder添加新的函数,使其能在beamsearch时按照新的order重排 '''

        def reorder_encoder_out(self, encoder_out, new_order):
            ''' encoder_out: [Tensor1,Tensor2...]'''
            if encoder_out["encoder_out"] is not None:
                num_tensors = len(encoder_out["encoder_out"])
                encoder_out["encoder_out"] = [encoder_out["encoder_out"][i].index_select(index=new_order, axis=0) for i
                                              in range(num_tensors)]
            if encoder_out["src_mask"] is not None:
                if len(encoder_out["src_mask"].shape) == 4:  # [bsz,1,1,src_len]
                    encoder_out["src_mask"] = encoder_out["src_mask"].index_select(index=new_order, axis=0)
                else:  # [streams,bsz,1,1,src_len]
                    encoder_out["src_mask"] = encoder_out["src_mask"].index_select(index=new_order, axis=1)

            return encoder_out

        encoder.reorder_encoder_out = types.MethodType(reorder_encoder_out, encoder)
        return encoder

    def forward_encoder(self, src_word):
        # 1.embed
        src_max_len = paddle.shape(src_word)[-1]
        src_mask = paddle.cast(src_word == self.pad_id, dtype=paddle.get_default_dtype()).unsqueeze(
            [-2, -3]) * -1e9  # [bsz,1,1,src_len]
        src_mask.stop_gradient = True
        # pos embed, 2,3...
        src_token_mask = paddle.cast(src_word != self.pad_id, dtype=src_word.dtype)
        src_pos = paddle.cumsum(src_token_mask, axis=-1) * src_token_mask + self.pad_id

        src_emb = self.src_word_embedding(src_word) * self.embed_scale + self.src_pos_embedding(src_pos)
        enc_input = F.dropout(src_emb, p=self.dropout, training=self.training) if self.dropout else src_emb
        # 2.wait-k
        with paddle.static.amp.fp16_guard():
            if self.stream and not self.training:  # TODO:以后支持流式文件训练
                # src [stream,bsz,len,dim]
                stream_len = src_word.shape[0]
                if (self.waitk - 1) >= stream_len or self.waitk == -1:  # 注意防止下面空！
                    # full sentence
                    encoder_outs = [self.encoder(enc_input[-1], src_mask=src_mask[-1])]
                else:
                    # waitk
                    encoder_outs = []
                    for i in range(self.waitk - 1,
                                   stream_len):  # self.waitk - 1<stream_len,否则为空,上一个分支为waitk-1>=stream_len
                        enc_output = self.encoder(enc_input[i], src_mask=src_mask[i])
                        encoder_outs.append(enc_output)
            else:
                if self.waitk >= src_max_len or self.waitk == -1:
                    # Full sentence
                    encoder_outs = [
                        self.encoder(
                            enc_input, src_mask=src_mask)
                    ]
                else:
                    # Wait-k policy
                    encoder_outs = []
                    for i in range(self.waitk, src_max_len + 1): # 等待k个token，译文永远比原文慢k个token
                        enc_output = self.encoder(
                            enc_input[:, :i, :],
                            src_mask=src_mask[:, :, :, :i])
                        encoder_outs.append(enc_output)

        return {
            "encoder_out": encoder_outs,  # tensor list
            "src_mask": src_mask,  # [bsz,1,1,src_len] / or [streams,bsz,1,1,src_len]
        }

    def forward_decoder(self, tgt_word, encoder_out=None, caches=None):
        # encoder output
        tgt_len = tgt_word.shape[-1]  # inference step num
        memory_mask, encoder_outs = None, None
        if encoder_out is not None:
            memory_mask = encoder_out["src_mask"]  # [bsz,1,1,src_len]
            encoder_outs = encoder_out["encoder_out"]
        # mask
        tgt_pad_mask = tgt_word == self.pad_id  # [bsz,tgt_len]
        if not caches:
            tgt_mask = paddle.tensor.triu(
                (paddle.ones(
                    (tgt_len, tgt_len),
                    dtype=paddle.get_default_dtype()) * -self.inf),
                1)  # [tgt_len,tgt_len]
            tgt_mask.stop_gradient = True
            # 加上pad mask
            if tgt_pad_mask.any():  # 如果tgt存在pad，需要对上三角再做mask
                tgt_mask = paddle.where(tgt_pad_mask.unsqueeze([1, 2]), paddle.to_tensor(-self.inf, dtype='float32'),
                                        tgt_mask)
        # for inference
        else:
            tgt_mask = paddle.cast(tgt_pad_mask, dtype=paddle.get_default_dtype()).unsqueeze(
                [1, 2]) * -1e9  # [bsz,1,1,tgt_len]

        # pos embed
        tgt_token_mask = paddle.cast(tgt_word != self.pad_id, dtype=tgt_word.dtype)
        tgt_pos = paddle.cumsum(tgt_token_mask, axis=-1) * tgt_token_mask + self.pad_id
        pos_embed = self.tgt_pos_embedding(tgt_pos)

        # slice word,embed and mask
        if not caches:
            memory = encoder_outs
        # for inference
        else:
            # take last column
            tgt_word = tgt_word[:, -1:]
            pos_embed = pos_embed[:, -1:]
            # take correct encoder outs
            step_num = tgt_len
            # if len(encoder_outs) == 1 or step_num > len(encoder_outs):  # full sentence
            if len(encoder_outs) == 1 or (step_num - 1) >= len(encoder_outs):  # full sentence
                memory = [encoder_outs[-1]]
                if self.stream:
                    memory_mask = memory_mask[-1]  # [streams,bsz,1,1,src_len]->[bsz,1,1,src_len]
                else:
                    memory_mask = memory_mask[:, :, :, :memory[0].shape[1]]
            else:  # len(encoder_outs)>1 and step<=len(encoder_outs)
                memory = [encoder_outs[step_num - 1]]
                if self.stream:
                    if len(encoder_outs) > len(memory_mask):  # 由于full长度会多次使用，会出现超过stream数，mask取最后一个，否则wait3时报错！
                        memory_mask = memory_mask[- 1]  # [streams,bsz,1,1,src_len]->[bsz,1,1,src_len]
                    else:
                        memory_mask = memory_mask[step_num - 1]  # [streams,bsz,1,1,src_len]->[bsz,1,1,src_len]
                else:
                    memory_mask = memory_mask[:, :, :, :memory[0].shape[1]]

        with paddle.static.amp.fp16_guard():
            tgt_emb = self.tgt_word_embedding(tgt_word) * self.embed_scale + pos_embed
            dec_input = F.dropout(
                tgt_emb, p=self.dropout,
                training=self.training) if self.dropout else tgt_emb
            if caches is None:
                output = self.decoder(dec_input, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, cache=None)
            else:
                output, new_caches = self.decoder(dec_input, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                                  cache=caches)

            logits = self.linear(output)
        attn = None
        return (logits, attn) if not caches else (logits, attn, new_caches)

    def forward(self, src_word, tgt_word):
        ''' for train '''
        encoder_out = self.forward_encoder(src_word)
        logits, attn = self.forward_decoder(tgt_word, encoder_out)
        return logits, None


def _create_transformer(variant, is_test, pretrained_path, args):
    model = SimultaneousTransformer(**args)
    mode = 'TRAIN' if not is_test else 'INFER'
    if is_test:
        model.eval()
    print(f'{mode} model {variant} created!')
    if pretrained_path is not None:
        state = paddle.load(pretrained_path)
        model.set_dict(state)
        print(f'Pretrained weight load from:{pretrained_path}!')
    k = args['waitk']
    print(f'================================ waitk={k} ================================\n')
    return model


def base_architecture(args):
    args["dropout"] = args.get("dropout", 0.1)
    args["d_model"] = args.get("d_model", 512)
    args["nheads"] = args.get("nheads", 8)
    args["dim_feedforward"] = args.get("dim_feedforward", 2048)
    args["encoder_layers"] = args.get("encoder_layers", 6)
    args["decoder_layers"] = args.get("decoder_layers", 6)
    return args


cfgs = ['src_vocab_size', 'tgt_vocab_size', 'waitk']
from ppseq.models import register_model_arch

@register_model_arch("transformer_simul_base")
def transformer_simul_base(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=8,
                      dim_feedforward=2048,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_simul_base', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_simul_big")
def transformer_simul_big(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=1024,
                      nheads=16,
                      dim_feedforward=4096,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_simul_big', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_simul_base_share")
def transformer_simul_base_share(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=8,
                      dim_feedforward=2048,
                      weight_sharing=True,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_base_share', is_test, pretrained_path, model_args)
    return model

