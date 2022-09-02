import types
import numpy as np
import os
import math
import paddle
from paddle.nn import layer, LayerList
import paddle.nn as nn
import paddle.distributed as dist
import paddle.nn.functional as F
from paddlenlp.transformers import PositionalEmbedding
# from .utils.xavier import xavier_uniform_with_gain
# from .utils.deep import deepnorm_init
import functools
import functools
import paddle.nn.initializer as I
from paddle.nn import MultiHeadAttention
import paddle.nn.initializer as I
from fastcore.all import patch_to, partial # 1.0
xavier_uniform_=I.XavierUniform()
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)


@patch_to(nn.Layer)
def apply(self, fn, name=""):
    for n, layer in self.named_children():
        nnmame = n if name == "" else name + "." + n
        layer.apply(fn, nnmame)

    fn(self, name)
    return self


def xavier_uniform_with_gain(tensor,gain):
    xavier_uniform_ = I.XavierUniform()
    xavier_uniform_._compute_fans = decorator(
        xavier_uniform_._compute_fans, gain=gain
    )
    xavier_uniform_(tensor)

def decorator(func, gain=1):
    @functools.wraps(func)
    def wrappper(*args, **kwargs):
        fan_in, fan_out = func(*args, **kwargs)
        return fan_in / (gain ** 2), fan_out / (gain ** 2)

    return wrappper


def xavier_normal_fn(weight,gain=1):
    ''' with torch init '''
    try:
        import torch
        import torch.nn as tnn
        w = torch.from_numpy(weight.numpy())
        w = tnn.init.xavier_normal_(w,gain=gain)
        weight.set_value(w.numpy())
    except ImportError as err:
        xavier_normal_gain = I.XavierNormal()
        xavier_normal_gain._compute_fans = decorator(
            xavier_normal_gain._compute_fans, gain=gain
        )
        xavier_normal_gain(weight)


def make_pad_zero(weight,pad_idx=1):
    weight.stop_gradient = True
    weight[pad_idx, :] = 0.
    weight.stop_gradient = False


def normal_fn_(weight,rand_norm=False,mean=0,std=1):
    '''
         weight: paddle weight
         rand_norm: normal random tensor , for embedding
    '''
    shape = weight.shape
    try:
        import torch
        import torch.nn as tnn
        if rand_norm:
            w = tnn.init.normal_(torch.randn(*shape),mean=mean,std=std)
        else:
            w = torch.from_numpy(weight.numpy())
            w = tnn.init.normal_(w,mean=mean,std=std)

        weight.set_value(w.numpy())
    except ImportError as err:
        from paddle.nn.initializer import Normal
        normal_ = Normal(mean=mean,std=std)
        normal_(weight)

def Embedding(num_embeddings, embedding_dim, padding_idx=1):
    m = nn.Embedding(num_embeddings,embedding_dim)
    # normalize
    normal_fn_(m.weight,rand_norm=True,mean=0,std=embedding_dim ** -0.5)
    # remove pad
    make_pad_zero(m.weight,padding_idx)
    return m

def Linear(in_features,out_features,bias=True):
    m = nn.Linear(in_features,out_features,bias_attr= bias)
    try:
        import torch
        import torch.nn as tnn
        m2 = tnn.Linear(in_features,out_features,bias=bias)
        m.weight.set_value(m2.weight.T.detach().numpy())
        if bias:
            m.bias.set_value(m2.bias.detach().numpy())
    except:
        pass
    return m


class MultiHeadAttentionWithInit(MultiHeadAttention):
    def __init__(self,*args,**kwargs):
        super(MultiHeadAttentionWithInit,self).__init__(*args,**kwargs)
        # out_proj's bias=0
        self.k_proj = Linear(self.kdim, self.embed_dim)
        self.v_proj = Linear(self.vdim, self.embed_dim)
        self.q_proj = Linear(self.embed_dim, self.embed_dim)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)

    def reset_paramaters(self):
        try:
            import torch
            import torch.nn as tnn
            kw=tnn.init.xavier_uniform_(torch.from_numpy(self.k_proj.weight.numpy()), gain=1 / math.sqrt(2))
            self.k_proj.weight.set_value(kw.numpy())
            del kw
            vw=tnn.init.xavier_uniform_(torch.from_numpy(self.v_proj.weight.numpy()), gain=1 / math.sqrt(2))
            self.v_proj.weight.set_value(vw.numpy())
            del vw
            qw=tnn.init.xavier_uniform_(torch.from_numpy(self.q_proj.weight.numpy()), gain=1 / math.sqrt(2))
            self.q_proj.weight.set_value(qw.numpy())
            del qw
            ow=tnn.init.xavier_uniform_(torch.from_numpy(self.out_proj.weight.numpy()))
            self.out_proj.weight.set_value(ow.numpy())
            del ow
        except ImportError as err:
            pass
        if self.out_proj.bias is not None:  # yes , out bias =0
            from paddle.nn.initializer import Constant
            zero_= Constant(value=0.0)
            zero_(self.out_proj.bias)


class PositionalEmbeddingLeanable(PositionalEmbedding):
    def __init__(self,pad_idx=1,learnable=False,*args,**kwargs):
        super(PositionalEmbeddingLeanable,self).__init__(*args,**kwargs)
        self.pad_idx= pad_idx
        self.learnable = learnable
        make_pad_zero(self.pos_encoder.weight,pad_idx)

    def forward(self, pos):
        pos_emb = self.pos_encoder(pos)
        if not self.learnable:
            pos_emb.stop_gradient = True
        return pos_emb


class Mlp(nn.Layer):
    def __init__(self,d_model,dim_feedforward,drop=0.,activation="relu"):
        super(Mlp, self).__init__()
        self.act=getattr(F,activation)
        self.dropout=nn.Dropout(p=drop)
        self.linear1=Linear(d_model,dim_feedforward)
        self.linear2=Linear(dim_feedforward,d_model)

    def forward(self,x):
        x=self.act(self.linear1(x))
        x=self.linear2(self.dropout(x))
        return x


class EncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,*args,**kwargs):
        super(EncoderLayer,self).__init__(*args,**kwargs)

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

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src=self.mlp(src)
        src = paddle.reshape(residual,shape=src.shape)  + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src if cache is None else (src, incremental_cache)



class DecoderLayer(nn.TransformerDecoderLayer):
    ''' modify mlp, caches'''
    def __init__(self,*args,**kwargs):
        super(DecoderLayer,self).__init__(*args,**kwargs)

        # rebuild self/corss attention and linear1/2
        d_model, nhead, dim_feedforward, activation = kwargs.get("d_model"),kwargs.get("nhead") , kwargs.get("dim_feedforward"), kwargs.get("activation")
        act_dropout = kwargs.get("dropout") if kwargs.get("act_dropout") is None else kwargs.get("act_dropout")
        attn_dropout = kwargs.get("dropout") if kwargs.get("attn_dropout") is None else kwargs.get("attn_dropout")

        del self.linear1,self.linear2,self.dropout
        self.self_attn = MultiHeadAttentionWithInit(embed_dim=d_model, num_heads=nhead, dropout=attn_dropout)
        self.cross_attn =MultiHeadAttentionWithInit(embed_dim=d_model, num_heads=nhead, dropout=attn_dropout)
        self.mlp = Mlp(d_model=d_model,
                      dim_feedforward=dim_feedforward,
                      drop=act_dropout,
                      activation=activation)


    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        # 1.self attn
        residual=tgt
        if self.normalize_before:
            tgt=self.norm1(tgt)
        if cache is None:
            tgt=self.self_attn(tgt,tgt,tgt,tgt_mask,None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt=residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt=self.norm1(tgt)

        # 2.cross attn
        residual=tgt
        attn_scores=None
        if self.normalize_before:
            tgt=self.norm2(tgt)
        if True:
        # if cache is None:
            if not self.cross_attn.need_weights:
                tgt=self.cross_attn(tgt,memory,memory,memory_mask,None)
            else:
                tgt, attn_scores = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else: # TODO: add static cache
            tgt,static_cache=self.cross_attn(tgt,memory,memory,memory_mask,
                                             cache[1])
        tgt=residual  + self.dropout2(tgt)
        if not self.normalize_before:
            tgt=self.norm2(tgt)

        # 3.ffn
        residual=tgt
        if self.normalize_before:
            tgt=self.norm3(tgt)
        tgt=self.mlp(tgt)
        tgt = paddle.reshape(residual,shape=tgt.shape)  + self.dropout3(tgt)
        if not self.normalize_before:
            tgt=self.norm3(tgt)

        output=(tgt,attn_scores)
        return output if cache is None else (output,(incremental_cache,))

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return (incremental_cache,)


class Encoder(nn.TransformerEncoder):
    ''' add encoder_layers'''
    def __init__(self,
                 embed_tokens,
                 embed_positions,
                 embed_scale,
                 pad_id=1,
                 dropout=0.,
                 encoder_layers=None,
                 *args,**kwargs):
        super(Encoder,self).__init__(*args,**kwargs)
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


class Decoder(nn.TransformerDecoder):
    def __init__(self,
                 embed_tokens,
                 embed_positions,
                 embed_scale,
                 pad_id=1,
                 dropout=0.,
                 decoder_layers=None,
                 *args,**kwargs):
        super(Decoder,self).__init__(*args,**kwargs)
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

        # encoder output
        tgt_len = tgt_tokens.shape[-1]  # inference step num

        # mask
        pad_mask = tgt_tokens == self.pad_id  # [bsz,tgt_len]
        if not caches:
            tgt_mask = paddle.tensor.triu(
                (paddle.ones(
                    (tgt_len, tgt_len),
                    dtype=paddle.get_default_dtype()) * -self.inf),
                1)  # [tgt_len,tgt_len]
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

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

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


class Transformer(nn.Layer):
    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 d_model=512,
                 encoder_layers=6,
                 decoder_layers=6,
                 nheads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 need_attn=True,
                 share_embed=False,
                 learnable_pos = False,
                 max_length=1024,
                 bos_id=0,
                 eos_id=2,
                 pad_id=1,
                 unk_id=3,
                 ):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.dropout = dropout
        self.learnable_pos = learnable_pos
        self.share_embed=share_embed
        self.need_attn=need_attn
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.d_model = d_model
        self.inf = 1e9
        # embedding
        src_embed_tokens = Embedding(num_embeddings=len(src_vocab),embedding_dim=d_model,padding_idx=pad_id)
        pos_length = max_length + self.pad_id + 1
        src_embed_positions = PositionalEmbeddingLeanable(pad_idx=pad_id,learnable=learnable_pos,emb_dim=d_model, max_length=pos_length)
        tgt_embed_positions = PositionalEmbeddingLeanable(pad_idx=pad_id,learnable=learnable_pos,emb_dim=d_model, max_length=pos_length)
        if share_embed:
            assert len(src_vocab) == len(tgt_vocab), (
                "Vocabularies in source and target should be same for weight sharing."
            )
            tgt_embed_tokens = src_embed_tokens
        else:
            tgt_embed_tokens = Embedding(num_embeddings=len(tgt_vocab), embedding_dim=d_model,
                                                padding_idx=pad_id)
        self.embed_scale = d_model ** 0.5

        # encoder
        encoder_layers_ls=self.build_layers(layer_name="EncoderLayer",
                                         d_model=d_model,
                                         nheads=nheads,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout,
                                         num_layers=encoder_layers)

        encoder_norm = None
        self.encoder=Encoder(
                            embed_tokens=src_embed_tokens,
                            embed_positions=src_embed_positions,
                            embed_scale=self.embed_scale,
                            pad_id = self.pad_id,
                            dropout = self.dropout,
                            encoder_layers = encoder_layers_ls,
                            # raw parameter
                            encoder_layer=encoder_layers_ls[0],
                            num_layers=self.encoder_layers,
                            norm=encoder_norm)

        # decoder
        decoder_layers_ls=self.build_layers(layer_name="DecoderLayer",
                                         d_model=d_model,
                                         nheads=nheads,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout,
                                         num_layers=decoder_layers)
        decoder_norm = None
        self.decoder=Decoder(
                            embed_tokens=tgt_embed_tokens,
                            embed_positions=tgt_embed_positions,
                            embed_scale=self.embed_scale,
                            pad_id=self.pad_id,
                            dropout=self.dropout,
                            decoder_layers=decoder_layers_ls,
                            # raw parameter
                            decoder_layer=decoder_layers_ls[0],
                            num_layers=self.decoder_layers,
                            norm=decoder_norm)

        self.output_projection = Linear(in_features=d_model,out_features=len(tgt_vocab),bias=False)
        normal_fn_(self.output_projection.weight, rand_norm=False, mean=0, std=d_model ** -0.5)

        # initialize parameters
        self.apply(self.reset_paramaters)


    def reset_paramaters(self,m,n): # model,name
            if isinstance(m,nn.Linear):
                if any(x in n for x in ["q_proj", "k_proj", "v_proj"]):
                    xavier_uniform_with_gain(m.weight,gain=2**-0.5)
                elif any(x in n for x in ["out_proj"]):
                    xavier_uniform_(m.weight)

    def build_layers(self,layer_name="EncoderLayer",d_model=512,nheads=8,dim_feedforward=2048,dropout=0.,
                     num_layers=6):
        assert layer_name in ["EncoderLayer","DecoderLayer"]
        layers_list=[]
        for i in range(num_layers):
            layer_i = eval(layer_name)(
                d_model=d_model,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_dropout=0.,
                act_dropout=0.,
                activation="relu",
                normalize_before=False,
                bias_attr=[True] *2 if layer_name=="EncoderLayer" else [True] *3)
            # need attention
            if layer_name == "DecoderLayer":
                layer_i.cross_attn.need_weights = self.need_attn

            layers_list.append(layer_i)

        return LayerList(layers_list)


    def forward_encoder(self, src_tokens):
        # 1. embed src tokens
        src_embed,src_mask=self.encoder.forward_embedding(src_tokens=src_tokens)

        # 2. forward
        encoder_out=self.encoder(src_embed,src_mask)
        return {
            "encoder_out": [encoder_out],  # tensor list
            "src_mask": src_mask,  # [bsz,1,1,src_len]
        }

    def forward_decoder(self, prev_tokens, encoder_out=None, caches=None):

        # 1. encoder output
        memory_mask, memory = None, None
        if encoder_out is not None:
            memory_mask = encoder_out["src_mask"]  # [bsz,1,1,src_len]
            memory = encoder_out["encoder_out"][0]

        # 2. embed tgt tokens
        tgt_embed,tgt_mask = self.decoder.forward_embedding(prev_tokens,caches=caches)


        with paddle.static.amp.fp16_guard():
            if caches is None:
                outputs = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, cache=None)
            else:
                outputs, new_caches = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                                  cache=caches)
            output, avg_attn_scores = outputs
            logits = self.output_projection(output)

        outputs = [logits,avg_attn_scores]
        if caches is not None:
            outputs.append(new_caches)
        return tuple(outputs)

    def forward(self, src_tokens, prev_tokens):
        ''' for train '''
        encoder_out = self.forward_encoder(src_tokens)
        logits, avg_attn_scores = self.forward_decoder(prev_tokens, encoder_out)

        return logits, avg_attn_scores


def _create_transformer(variant, is_test, pretrained_path, args):
    model = Transformer(**args)
    mode = 'TRAIN' if not is_test else 'INFER'
    if is_test:
        model.eval()
    print(f'{mode} model {variant} created!')
    if pretrained_path is not None:
        state = paddle.load(pretrained_path)
        model.set_dict(state)
        print(f'Pretrained weight load from:{pretrained_path}!')
    return model

def base_architecture(args):
    args["dropout"] = args.get("dropout", 0.1)
    args["d_model"] = args.get("d_model", 512)
    args["nheads"] = args.get("nheads", 8)
    args["dim_feedforward"] = args.get("dim_feedforward", 2048)
    args["encoder_layers"] = args.get("encoder_layers", 6)
    args["decoder_layers"] = args.get("decoder_layers", 6)
    return args

cfgs = ['src_vocab', 'tgt_vocab']


def transformer_iwslt_de_en(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=4,
                      dim_feedforward=1024,
                      share_embed=True,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_iwslt_de_en', is_test, pretrained_path, model_args)
    return model


def transformer_base(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=8,
                      dim_feedforward=2048,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_base', is_test, pretrained_path, model_args)
    return model



def transformer_big(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=1024,
                      nheads=16,
                      dim_feedforward=4096,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_big', is_test, pretrained_path, model_args)
    return model



