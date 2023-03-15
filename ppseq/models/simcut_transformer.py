'''
把simcut添加到参数
simcut_p
'''
import types
import paddle
from ppseq.models.transformer import Transformer
import paddle.nn.functional as F
from functools import partial

class TransformerWithSimcut(Transformer):
    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 **kwargs
                 ):
        super(TransformerWithSimcut, self).__init__(src_vocab, tgt_vocab, **kwargs)
        self.encoder = self.modify_encoder(self.encoder)
        self.decoder = self.modify_decoder(self.decoder)

    # 修改forward embedding
    def modify_encoder(self, encoder):
        def forward_embedding(self, src_tokens, simcut_p=None):
            pad_mask = paddle.cast(src_tokens == self.pad_id, dtype=paddle.get_default_dtype()).unsqueeze(
                [-2, -3]) * -1e9  # [bsz,1,1,src_len]
            pad_mask.stop_gradient = True

            token_embed = self.embed_tokens(src_tokens)
            # token simcut
            if simcut_p is not None:
                bsz, seq = token_embed.shape[:2]
                token_embedding_mask = paddle.cast(paddle.rand([bsz, seq]) < (1 - simcut_p), dtype=token_embed.dtype)
                token_embedding_mask[:, -1] = 1  # Do not mask eos token
                # special_tokens = [0, 1, 2, 3]
                # for tok in special_tokens:
                #     sp_mask = src_tokens==tok
                #     token_embedding_mask[sp_mask]=1
                token_embed = paddle.multiply(token_embedding_mask.unsqueeze(-1), token_embed)

            token_embed = token_embed * self.embed_scale
            # postion embedding
            token_mask = paddle.cast(src_tokens != self.pad_id, dtype=src_tokens.dtype)
            src_pos = paddle.cumsum(token_mask, axis=-1) * token_mask + self.pad_id
            token_embed = token_embed + self.embed_positions(src_pos)
            # dropout
            token_embed = F.dropout(token_embed, p=self.dropout,
                                    training=self.training) if self.dropout else token_embed

            return token_embed, pad_mask

        encoder.forward_embedding = types.MethodType(forward_embedding, encoder)

        return encoder

    def modify_decoder(self, decoder):
        def forward_embedding(self, tgt_tokens, caches=None, simcut_p=None):
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
                # 加上pad mask
                if pad_mask.any():  # 如果tgt存在pad，需要对上三角再做mask
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
            if simcut_p is not None:
                bsz, seq = token_embed.shape[:2]
                cut_mask = paddle.cast(paddle.rand([bsz, seq]) < (1 - simcut_p), dtype=token_embed.dtype)
                cut_mask[:, 0] = 1  # Do not mask bos token
                # special_tokens = [0, 1, 2, 3] # bos pad eos unk
                # for tok in special_tokens:
                #     sp_mask = tgt_tokens==tok
                #     cut_mask[sp_mask]=1
                token_embed = paddle.multiply(cut_mask.unsqueeze(-1), token_embed)

            token_embed = token_embed * self.embed_scale + pos_embed

            token_embed = F.dropout(
                token_embed, p=self.dropout,
                training=self.training) if self.dropout else token_embed
            return token_embed, tgt_mask

        decoder.forward_embedding = types.MethodType(forward_embedding, decoder)

        return decoder


    def forward_encoder(self, src_tokens ,simcut_p=None):
        # 1. embed src tokens
        src_embed,src_mask=self.encoder.forward_embedding(src_tokens=src_tokens,simcut_p=simcut_p)

        # 2. forward
        encoder_out=self.encoder(src_embed,src_mask)
        return {
            "encoder_out": [encoder_out],  # tensor list
            "src_mask": src_mask,  # [bsz,1,1,src_len]
        }

    def forward_decoder(self, prev_tokens, encoder_out=None, caches=None ,simcut_p=None):

        # 1. encoder output
        memory_mask, memory = None, None
        if encoder_out is not None:
            memory_mask = encoder_out["src_mask"]  # [bsz,1,1,src_len]
            memory = encoder_out["encoder_out"][0]

        # 2. embed tgt tokens
        # print(self.decoder.forward_embedding.__code__.co_varnames[:self.decoder.forward_embedding.__code__.co_argcount])
        tgt_embed,tgt_mask = self.decoder.forward_embedding(prev_tokens,caches=caches,simcut_p=simcut_p)


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

def _create_transformer(variant, is_test, pretrained_path, args):
    model = TransformerWithSimcut(**args)
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
@register_model_arch("simcut_iwslt_de_en")
def simcut_iwslt_de_en(is_test=False, pretrained_path=None, **kwargs):
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
    model = _create_transformer('simcut_iwslt_de_en', is_test, pretrained_path, model_args)
    return model

@register_model_arch("simcut_iwslt_de_en_norm")
def simcut_iwslt_de_en_norm(is_test=False, pretrained_path=None, **kwargs):
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
    model = _create_transformer('simcut_iwslt_de_en_norm', is_test, pretrained_path, model_args)
    return model

@register_model_arch("simcut_base")
def simcut_base(is_test=False, pretrained_path=None, **kwargs):
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


@register_model_arch("simcut_big")
def simcut_big(is_test=False, pretrained_path=None, **kwargs):
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
    model = _create_transformer('simcut_big', is_test, pretrained_path, model_args)
    return model



