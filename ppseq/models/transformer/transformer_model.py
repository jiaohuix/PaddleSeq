import math
import paddle.nn.initializer as init
import paddle
from paddle.nn import LayerList
import paddle.nn as nn
from functools import partial
from ppseq.modules import PositionalEmbeddingLeanable,Linear,LayerDropList,Embedding
from ppseq.modules.torch_utils import normal_fn_,make_pad_zero
from ppseq.models.transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)
from ppseq.modules.deepnet import xavier_uniform_,xavier_uniform_with_gain,deepnorm_init
'''
22/9/29 NEW ADD:
1.deepnorm √
2.activation√
3.layerdrop√
'''

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
                 attn_dropout=0.,
                 act_dropout=0.,
                 need_attn=True,
                 share_embed=False,
                 share_all=False,
                 use_deepnorm=False,
                 learnable_pos = False,
                 activation = "relu",
                 encoder_layerdrop=0.,
                 decoder_layerdrop=0.,
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
        self.attn_dropout = attn_dropout
        self.act_dropout = act_dropout
        self.share_embed = share_embed
        self.use_deepnorm = use_deepnorm
        self.learnable_pos = learnable_pos
        self.activation = activation
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.need_attn=need_attn
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
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
            tgt_embed_tokens = Embedding(num_embeddings=len(tgt_vocab), embedding_dim=d_model, padding_idx=pad_id)
        self.embed_scale = d_model ** 0.5

        # encoder
        encoder_layers_ls=self.build_layers(layer_name="TransformerEncoderLayer")

        encoder_norm = None
        self.encoder=TransformerEncoder(
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
        decoder_layers_ls=self.build_layers(layer_name="TransformerDecoderLayer")

        decoder_norm = None
        self.decoder=TransformerDecoder(
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

        if share_embed and share_all: # share_decoder_input_output_embed
            self.output_projection = lambda x: paddle.matmul(
                    x=x, y=self.decoder.embed_tokens.weight, transpose_y=True)
        else:
            self.output_projection = Linear(in_features=d_model,out_features=len(tgt_vocab),bias=False)
            normal_fn_(self.output_projection.weight, rand_norm=False, mean=0, std=d_model ** -0.5)

        # initialize parameters
        if self.use_deepnorm:
            deepnorm_init_fn=partial(deepnorm_init,N=self.encoder_layers,M=self.decoder_layers)
            self.apply(deepnorm_init_fn)
        # else:
        #     self.apply(self.reset_paramaters)

    def reset_paramaters(self,m,n): # model,name   # initialize attention
        if isinstance(m,nn.Linear):
            if any(x in n for x in ["q_proj", "k_proj", "v_proj"]):
                xavier_uniform_with_gain(m.weight,gain=2**-0.5)
                if m.bias is not None:
                    in_features= m.weight.shape[1]
                    Uniform_ = init.Uniform(-1 / math.sqrt(in_features),
                                             1 / math.sqrt(in_features))
                    Uniform_(m.bias)

            elif any(x in n for x in ["out_proj"]):
                xavier_uniform_(m.weight)

    def build_layers(self,layer_name="TransformerEncoderLayer"):
        assert layer_name in ["TransformerEncoderLayer","TransformerDecoderLayer"]
        num_layers = self.encoder_layers if layer_name=="TransformerEncoderLayer" else self.decoder_layers
        layers_list=[]
        for i in range(num_layers):
            layer_i = eval(layer_name)(
                use_deepnorm = self.use_deepnorm,
                d_model = self.d_model,
                nhead = self.nheads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                attn_dropout = self.attn_dropout,
                act_dropout = self.act_dropout,
                activation = self.activation,
                normalize_before = False,
                bias_attr=[True] * 2 if layer_name=="TransformerEncoderLayer" else [True] *3)
            # need attention
            if layer_name == "TransformerDecoderLayer":
                layer_i.cross_attn.need_weights = self.need_attn

            layers_list.append(layer_i)

        if layer_name == "TransformerEncoderLayer" and self.encoder_layerdrop>0:
            return LayerDropList(p=self.encoder_layerdrop,layers=layers_list)
        elif layer_name == "TransformerDecoderLayer" and self.decoder_layerdrop>0:
            return LayerDropList(p=self.decoder_layerdrop,layers=layers_list)
        else:
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
    args["attn_dropout"] = args.get("attn_dropout", 0.)
    args["act_dropout"] = args.get("act_dropout", 0.)
    args["d_model"] = args.get("d_model", 512)
    args["nheads"] = args.get("nheads", 8)
    args["dim_feedforward"] = args.get("dim_feedforward", 2048)
    args["encoder_layers"] = args.get("encoder_layers", 6)
    args["decoder_layers"] = args.get("decoder_layers", 6)
    args["activation"] = args.get("activation", "relu")
    # layerdrop
    args["encoder_layerdrop"] = args.get("encoder_layerdrop", 0)
    args["decoder_layerdrop"] = args.get("decoder_layerdrop", 0)

    return args

cfgs = ['src_vocab', 'tgt_vocab']

from ppseq.models import register_model_arch
@register_model_arch("transformer_iwslt_de_en")
def transformer_iwslt_de_en(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=4,
                      dim_feedforward=1024,
                      share_embed=True,
                      share_all = True,
                      use_deepnorm=False,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_iwslt_de_en', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_iwslt_de_en_norm")
def transformer_iwslt_de_en_norm(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=4,
                      dim_feedforward=1024,
                      share_embed=True,
                      share_all = True,
                      use_deepnorm=True,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_iwslt_de_en_norm', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_base")
def transformer_base(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=8,
                      dim_feedforward=2048,
                      share_embed=False,
                      use_deepnorm=False,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_base', is_test, pretrained_path, model_args)
    return model


@register_model_arch("transformer_big")
def transformer_big(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=1024,
                      nheads=16,
                      dim_feedforward=4096,
                      share_embed=False,
                      use_deepnorm=False,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_big', is_test, pretrained_path, model_args)
    return model



