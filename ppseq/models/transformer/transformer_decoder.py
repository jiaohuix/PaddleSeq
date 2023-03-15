import paddle
import paddle.nn as nn
from paddle.nn import  LayerList
import paddle.nn.functional as F
from ppseq.modules import Mlp,MultiHeadAttentionWithInit
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
)


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    ''' modify mlp, caches'''
    def __init__(self,
                 use_deepnorm=False,
                 no_encoder_attn=False,
                 *args,**kwargs):
        super(TransformerDecoderLayer,self).__init__(*args,**kwargs)
        self.no_encoder_attn = no_encoder_attn
        # for deepnorm
        self.alpha = 1.0

        # rebuild self/corss attention and linear1/2
        d_model, nhead, dim_feedforward, activation = kwargs.get("d_model"),kwargs.get("nhead") , kwargs.get("dim_feedforward"), kwargs.get("activation")
        act_dropout = kwargs.get("dropout") if kwargs.get("act_dropout") is None else kwargs.get("act_dropout")
        attn_dropout = kwargs.get("dropout") if kwargs.get("attn_dropout") is None else kwargs.get("attn_dropout")

        del self.linear1,self.linear2,self.dropout
        self.self_attn = MultiHeadAttentionWithInit(embed_dim=d_model, num_heads=nhead, dropout=attn_dropout)
        self.cross_attn =MultiHeadAttentionWithInit(embed_dim=d_model, num_heads=nhead, dropout=attn_dropout)
        if self.no_encoder_attn:
            del self.cross_attn,self.norm2
        self.mlp = Mlp(d_model=d_model,
                      dim_feedforward=dim_feedforward,
                      drop=act_dropout,
                      activation=activation)


    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None, cache=None):
        # 1.self attn
        residual=tgt
        if self.normalize_before:
            tgt=self.norm1(tgt)
        if cache is None:
            tgt=self.self_attn(tgt,tgt,tgt,tgt_mask,None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt=residual * self.alpha + self.dropout1(tgt)
        if not self.normalize_before:
            tgt=self.norm1(tgt)

        # 2.cross attn
        attn_scores = None
        if not self.no_encoder_attn:
            residual = tgt
            if self.normalize_before:
                tgt = self.norm2(tgt)
            if True:
                # if cache is None:
                if not self.cross_attn.need_weights:
                    tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
                else:
                    tgt, attn_scores = self.cross_attn(tgt, memory, memory, memory_mask, None)
            else:  # TODO: add static cache
                tgt, static_cache = self.cross_attn(tgt, memory, memory, memory_mask,
                                                    cache[1])
            tgt = residual * self.alpha + self.dropout2(tgt)
            if not self.normalize_before:
                tgt = self.norm2(tgt)

        # 3.ffn
        residual=tgt
        if self.normalize_before:
            tgt=self.norm3(tgt)
        tgt=self.mlp(tgt)
        tgt = paddle.reshape(residual,shape=tgt.shape) * self.alpha + self.dropout3(tgt)
        if not self.normalize_before:
            tgt=self.norm3(tgt)

        output=(tgt,attn_scores)
        return output if cache is None else (output,(incremental_cache,))

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return (incremental_cache,)


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self,
                 embed_tokens,
                 embed_positions,
                 embed_scale,
                 pad_id=1,
                 dropout=0.,
                 decoder_layers=None,
                 *args,**kwargs):
        super(TransformerDecoder,self).__init__(*args,**kwargs)
        self.embed_tokens=embed_tokens
        self.embed_positions=embed_positions
        self.embed_scale=embed_scale
        self.inf = 1e9
        self.pad_id=pad_id
        self.dropout=dropout
        if decoder_layers is not None and isinstance(decoder_layers, LayerList):
            self.layers = decoder_layers

    def gen_caches(self, memory):
        ''' [(increment_cache,),...] '''
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

    def forward_embedding(self,tgt_tokens,caches=None):
        diagonal = 1
        # encoder output
        tgt_len = tgt_tokens.shape[-1]  # inference step num

        # mask
        pad_mask = tgt_tokens == self.pad_id  # [bsz,tgt_len]
        if not caches:
            tgt_mask = paddle.tensor.triu(
                (paddle.ones(
                    (tgt_len, tgt_len),
                    dtype=paddle.get_default_dtype()) * -self.inf),
                diagonal=diagonal)  # [tgt_len,tgt_len]
            tgt_mask.stop_gradient = True
            # add tgt pad mask
            if pad_mask.any():
                tgt_mask = paddle.where(pad_mask.unsqueeze([1, 2]), paddle.to_tensor(-self.inf, dtype='float32'),
                                        tgt_mask)
        # for inference
        else:
            tgt_mask = paddle.cast(pad_mask, dtype=paddle.get_default_dtype()).unsqueeze(
                [1, 2]) * -1e9  # [bsz,1,1,tgt_len]

        # pos embed
        token_mask = paddle.cast(tgt_tokens != self.pad_id, dtype=tgt_tokens.dtype)
        tgt_pos = paddle.cumsum(token_mask, axis=-1) * token_mask + self.pad_id
        pos_embed = self.embed_positions(tgt_pos)

        if caches is not None:
            # take last column
            tgt_tokens = tgt_tokens[:, -1:]
            pos_embed = pos_embed[:, -1:]

        token_embed = self.embed_tokens(tgt_tokens)
        token_embed = token_embed * self.embed_scale + pos_embed

        token_embed = F.dropout(
            token_embed, p=self.dropout,
            training=self.training) if self.dropout else token_embed
        return token_embed, tgt_mask

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None, cache=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype) if memory_mask is not None else None

        output = tgt
        new_caches = []
        avg_attn_scores = None
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

            # get attention scores
            output,attn_scores = output
            if not mod.no_encoder_attn:
                if mod.cross_attn.need_weights:
                    attn_scores = attn_scores / self.num_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores += attn_scores

                    # average probabilities over heads
                    # [bsz,heads,tgt_size,src_size] => [bsz,tgt_size,src_size]
                    if i==(self.num_layers-1):
                        avg_attn_scores =avg_attn_scores.mean(axis=1)

        if self.norm is not None:
            output = self.norm(output)
        outputs = (output,avg_attn_scores)
        return outputs if cache is None else (outputs, new_caches)


