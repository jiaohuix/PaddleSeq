import paddle
from paddle.nn import  LayerList
import paddle.nn as nn
import paddle.nn.functional as F
from ppseq.modules import Mlp,MultiHeadAttentionWithInit
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
                 use_deepnorm=False,
                 *args,**kwargs):
        super(TransformerEncoderLayer,self).__init__(*args,**kwargs)
        # for deepnorm
        self.alpha = 1.0

        # rebuild self_attention and linear1/2
        d_model, nhead, dim_feedforward, activation = kwargs.get("d_model"),kwargs.get("nhead") , kwargs.get("dim_feedforward"), kwargs.get("activation")
        act_dropout = kwargs.get("dropout") if kwargs.get("act_dropout") is None else kwargs.get("act_dropout")
        attn_dropout = kwargs.get("dropout") if kwargs.get("attn_dropout") is None else kwargs.get("attn_dropout")

        del self.linear1,self.linear2,self.dropout
        self.self_attn = MultiHeadAttentionWithInit(embed_dim=d_model, num_heads=nhead, dropout=attn_dropout)

        self.mlp = Mlp(d_model=d_model,
                      dim_feedforward=dim_feedforward,
                      drop=act_dropout,
                      activation=activation)

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        # Add cache for encoder for the usage like UniLM
        if cache is None: # encoder just forward one time, do not need cache
            src = self.self_attn(src, src, src, src_mask)
        else:
            src, incremental_cache = self.self_attn(src, src, src, src_mask,
                                                    cache)

        src = residual * self.alpha  + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src=self.mlp(src)
        src = paddle.reshape(residual,shape=src.shape) * self.alpha + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src if cache is None else (src, incremental_cache)


class TransformerEncoder(nn.TransformerEncoder):
    ''' add encoder_layers'''
    def __init__(self,
                 embed_tokens,
                 embed_positions,
                 embed_scale,
                 pad_id=1,
                 dropout=0.,
                 encoder_layers=None,
                 *args,**kwargs):
        super(TransformerEncoder,self).__init__(*args,**kwargs)
        self.embed_tokens=embed_tokens
        self.embed_positions=embed_positions
        self.embed_scale=embed_scale
        self.pad_id=pad_id
        self.dropout=dropout
        if encoder_layers is not None and isinstance(encoder_layers,LayerList):
            self.layers=encoder_layers

    def reorder_encoder_out(self, encoder_out, new_order):
        ''' encoder_out: [Tensor1,Tensor2...]'''
        if encoder_out["encoder_out"] is not None:
            num_tensors = len(encoder_out["encoder_out"])
            encoder_out["encoder_out"] = [encoder_out["encoder_out"][i].index_select(index=new_order, axis=0) for i
                                          in range(num_tensors)]
        if encoder_out["src_mask"] is not None:
                encoder_out["src_mask"] = encoder_out["src_mask"].index_select(index=new_order, axis=0) # [bsz,1,1,src_len]

        return encoder_out

    def forward_embedding(self,src_tokens):
        pad_mask = paddle.cast(src_tokens == self.pad_id, dtype=paddle.get_default_dtype()).unsqueeze(
                              [-2, -3]) * -1e9  # [bsz,1,1,src_len]
        pad_mask.stop_gradient = True

        token_embed = self.embed_tokens(src_tokens)

        token_embed = token_embed * self.embed_scale
        # postion embedding
        token_mask = paddle.cast(src_tokens != self.pad_id, dtype=src_tokens.dtype)
        src_pos = paddle.cumsum(token_mask, axis=-1) * token_mask + self.pad_id
        token_embed = token_embed + self.embed_positions(src_pos)
        # dropout
        token_embed = F.dropout(token_embed, p=self.dropout, training=self.training) if self.dropout else token_embed

        return token_embed,pad_mask

