import paddle
from paddle.nn import LayerList
import paddle.nn as nn
from functools import partial
from ppseq.modules import PositionalEmbeddingLeanable,Linear,LayerDropList
from ppseq.modules.torch_utils import normal_fn_,make_pad_zero
from ppseq.models.transformer import (
    TransformerDecoderLayer,
    TransformerDecoder,
)
from ppseq.modules.deepnet import xavier_uniform_,xavier_uniform_with_gain,deepnorm_init
'''
存在问题：
1、预测时候如何设置prefix，要修改future mask吗，让前prefix个互相看得见
'''

def Embedding(num_embeddings, embedding_dim, padding_idx=1):
    m = nn.Embedding(num_embeddings,embedding_dim)
    # normalize
    normal_fn_(m.weight,rand_norm=True,mean=0,std=embedding_dim ** -0.5)
    # remove pad
    make_pad_zero(m.weight,padding_idx)
    return m


class TransformerLanguageModel(nn.Layer):
    def __init__(self,
                 src_vocab,
                 d_model=512,
                 decoder_layers=6,
                 nheads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attn_dropout=0.,
                 act_dropout=0.,
                 need_attn=True,
                 share_embed=False, # share_decoder_input_output_embed
                 use_deepnorm=False,
                 learnable_pos = False,
                 activation = "relu",
                 decoder_layerdrop=0.,
                 max_length=1024,
                 bos_id=0,
                 eos_id=2,
                 pad_id=1,
                 unk_id=3,
                 ):
        super(TransformerLanguageModel, self).__init__()
        self.src_vocab = src_vocab
        self.src_vocab_size = len(src_vocab)
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.act_dropout = act_dropout
        self.share_embed=share_embed
        self.use_deepnorm = use_deepnorm
        self.learnable_pos = learnable_pos
        self.activation = activation
        self.decoder_layerdrop = decoder_layerdrop
        self.need_attn=need_attn
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.no_encoder_attn = True
        self.inf = 1e9
        # embedding
        pos_length = max_length + self.pad_id + 1
        embed_tokens = Embedding(num_embeddings=self.src_vocab_size, embedding_dim=d_model, padding_idx=pad_id)
        embed_positions = PositionalEmbeddingLeanable(pad_idx=pad_id,learnable=learnable_pos,emb_dim=d_model, max_length=pos_length)
        self.embed_scale = d_model ** 0.5

        # decoder
        decoder_layers_ls=self.build_decoder_layers()
        decoder_norm = None
        self.decoder=TransformerDecoder(
                            embed_tokens=embed_tokens,
                            embed_positions=embed_positions,
                            embed_scale=self.embed_scale,
                            pad_id=self.pad_id,
                            dropout=self.dropout,
                            decoder_layers=decoder_layers_ls,
                            # raw parameter
                            decoder_layer=decoder_layers_ls[0],
                            num_layers=self.decoder_layers,
                            norm=decoder_norm)

        if share_embed: # share_decoder_input_output_embed
            self.output_projection = lambda x: paddle.matmul(
                    x=x, y=self.decoder.embed_tokens.weight, transpose_y=True)
        else:
            self.output_projection = Linear(in_features=d_model,out_features=self.src_vocab_size,bias=False)
            normal_fn_(self.output_projection.weight, rand_norm=False, mean=0, std=d_model ** -0.5)

        # initialize parameters
        if self.use_deepnorm:
            deepnorm_init_fn=partial(deepnorm_init,N=self.encoder_layers,M=self.decoder_layers)
            self.apply(deepnorm_init_fn)
        else:
            self.apply(self.reset_paramaters)

    def reset_paramaters(self,m,n): # model,name
            if isinstance(m,nn.Linear):
                if any(x in n for x in ["q_proj", "k_proj", "v_proj"]):
                    xavier_uniform_with_gain(m.weight,gain=2**-0.5)
                elif any(x in n for x in ["out_proj"]):
                    xavier_uniform_(m.weight)

    def build_decoder_layers(self):
        layers_list=[]
        for i in range(self.decoder_layers):
            layer_i = TransformerDecoderLayer(
                use_deepnorm = self.use_deepnorm,
                no_encoder_attn = self.no_encoder_attn, # no cross attention
                d_model = self.d_model,
                nhead = self.nheads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                attn_dropout = self.attn_dropout,
                act_dropout = self.act_dropout,
                activation = self.activation,
                normalize_before = False,
                bias_attr=[True] *3)
            # need attention
            if not self.no_encoder_attn:
                layer_i.cross_attn.need_weights = self.need_attn
            layers_list.append(layer_i)

        if self.decoder_layerdrop>0:
            return LayerDropList(p=self.decoder_layerdrop,layers=layers_list)
        else:
            return LayerList(layers_list)

    # prefix tokens 设置windows size，
    def forward_decoder(self, prev_tokens, caches=None):

        # 1. no encoder output

        # 2. embed prev tokens
        prev_embed,prev_mask = self.decoder.forward_embedding(prev_tokens,caches=caches)

        with paddle.static.amp.fp16_guard():
            if caches is None:
                outputs = self.decoder(prev_embed,tgt_mask=prev_mask,cache=None)
            else:
                outputs, new_caches = self.decoder(prev_embed, tgt_mask=prev_mask, cache=caches)
            output, avg_attn_scores = outputs
            logits = self.output_projection(output)

        outputs = [logits,avg_attn_scores]
        if caches is not None:
            outputs.append(new_caches)
        return tuple(outputs)

    def forward(self, src_tokens):
        logits, avg_attn_scores = self.forward_decoder(src_tokens)

        return logits, avg_attn_scores

def _create_transformer(variant, is_test, pretrained_path, args):
    model = TransformerLanguageModel(**args)
    mode = 'TRAIN' if not is_test else 'INFER'
    if is_test:
        model.eval()
    print(f'{mode} model {variant} created!')
    if pretrained_path is not None:
        state = paddle.load(pretrained_path)
        model.set_dict(state)
        print(f'Pretrained weight load from:{pretrained_path}!')
    return model

def base_lm_architecture(args):
    args["dropout"] = args.get("dropout", 0.1)
    args["attn_dropout"] = args.get("attn_dropout", 0.)
    args["act_dropout"] = args.get("act_dropout", 0.)
    args["d_model"] = args.get("d_model", 512)
    args["nheads"] = args.get("nheads", 8)
    args["dim_feedforward"] = args.get("dim_feedforward", 2048)
    args["decoder_layers"] = args.get("decoder_layers", 6)
    args["activation"] = args.get("activation", "relu")
    # layerdrop
    args["decoder_layerdrop"] = args.get("decoder_layerdrop", 0)

    return args

cfgs = ['src_vocab']
from ppseq.models import register_model_arch

# GPT-1
@register_model_arch("transformer_lm_base")
def transformer_lm_base(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=12,
                      d_model=512,
                      nheads=8,
                      dim_feedforward=2048,
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_base', is_test, pretrained_path, model_args)
    return model


@register_model_arch("transformer_lm_big")
def transformer_lm_big(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=12,
                      d_model=1024,
                      nheads=16,
                      dim_feedforward=4096,
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_big', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_lm_gpt")
def transformer_lm_gpt(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=12,
                      d_model=768,
                      nheads=12,
                      dim_feedforward=3072,
                      attn_dropout=0.1,
                      activation="gelu",
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_gpt', is_test, pretrained_path, model_args)
    return model

# GPT-2
@register_model_arch("transformer_lm_gpt2_small")
def transformer_lm_gpt2_small(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=24,
                      d_model=1024,
                      nheads=16,
                      dim_feedforward=4096,
                      attn_dropout=0.1,
                      activation="gelu",
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_gpt2_small', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_lm_gpt2_tiny")
def transformer_lm_gpt2_tiny(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=2,
                      d_model=64,
                      nheads=1,
                      dim_feedforward=64,
                      attn_dropout=0.1,
                      activation="gelu",
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_gpt2_tiny', is_test, pretrained_path, model_args)
    return model

# GPT-3 used learned positional embeddings, rather than sinusoidal
# norm初始化
@register_model_arch("transformer_lm_gpt3_small")
def transformer_lm_gpt3_small(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=12,
                      d_model=768,
                      nheads=12,
                      dim_feedforward=3072,
                      attn_dropout=0.,
                      learnable_pos=True,
                      share_embed=True,
                      activation="gelu",
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_gpt3_small', is_test, pretrained_path, model_args)
    return model


@register_model_arch("transformer_lm_gpt3_medium")
def transformer_lm_gpt3_medium(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=24,
                      d_model=1024,
                      nheads=16,
                      dim_feedforward=4096,
                      attn_dropout=0.,
                      learnable_pos=True,
                      share_embed=True,
                      activation="gelu",
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_gpt3_medium', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_lm_gpt3_large")
def transformer_lm_gpt3_large(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_layers=24,
                      d_model=1536,
                      nheads=16,
                      dim_feedforward=6144,
                      attn_dropout=0.,
                      learnable_pos=True,
                      share_embed=True,
                      activation="gelu",
                      **kwargs)
    model_args = base_lm_architecture(model_args)
    model = _create_transformer('transformer_lm_gpt3_medium', is_test, pretrained_path, model_args)
    return model