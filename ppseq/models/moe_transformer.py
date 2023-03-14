'''
0.验证以前的错误： 1.linear初始 2.adamw 3.out proj share
1.将moe替换mlp √
2.结构验证。 √
3.deepnorm似乎没传入。
'''
import paddle
from paddle.incubate.distributed.models.moe import MoELayer,GShardGate,SwitchGate
from ppseq.modules import Mlp
from paddle.nn import LayerList
from functools import partial
from ppseq.models.transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Transformer
)
from paddle.distributed import fleet
from ppseq.modules.deepnet import xavier_uniform_,xavier_uniform_with_gain,deepnorm_init


def build_moe_layer(d_model,dim_feedforward,num_experts=8,gate_type="gshard",top_k=2,drop=0.,activation="relu"):
    moe_group = paddle.distributed.new_group(list(range(fleet.worker_num())))
    mp_group = paddle.distributed.new_group(list(range(fleet.worker_num())))
    experts_list = LayerList()
    for expi in range(num_experts):
        exp_layer = Mlp(d_model, dim_feedforward // top_k,drop=drop,activation=activation)
        experts_list.append(exp_layer)

    gate_config = {
        "type": gate_type,
        "top_k": top_k,
    }

    moeLayer = MoELayer(d_model=d_model,
                        experts=experts_list,
                        gate=gate_config,
                        moe_group=moe_group,
                        mp_group=mp_group,
                        recompute_interval=0)
    return moeLayer



class MoETransformer(Transformer):
    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 gate_type="gshard",
                 top_k=2,
                 enc_moe_experts = [8,1] * 3,
                 dec_moe_experts = [8,1] * 3,
                 **kwargs
                 ):
        super(MoETransformer, self).__init__(src_vocab, tgt_vocab, **kwargs)
        self.gate_type = gate_type
        self.top_k = top_k
        self.enc_moe_experts = enc_moe_experts
        self.dec_moe_experts = dec_moe_experts

        # encoder

        encoder_layers_ls=self.build_moe_layers(layer_name="TransformerEncoderLayer",
                                         num_layers=len(enc_moe_experts),
                                         moe_experts=dec_moe_experts)

        encoder_norm = None
        self.encoder=TransformerEncoder(
                            embed_tokens= self.encoder.embed_tokens,
                            embed_positions= self.encoder.embed_positions,
                            embed_scale=self.embed_scale,
                            pad_id = self.pad_id,
                            dropout = self.dropout,
                            encoder_layers = encoder_layers_ls,
                            # raw parameter
                            encoder_layer=encoder_layers_ls[0],
                            num_layers=self.encoder_layers,
                            norm=encoder_norm)

        # decoder
        decoder_layers_ls=self.build_moe_layers(layer_name="TransformerDecoderLayer",
                                         num_layers=len(dec_moe_experts),
                                         moe_experts=dec_moe_experts)
        decoder_norm = None
        self.decoder=TransformerDecoder(
                            embed_tokens=self.decoder.embed_tokens,
                            embed_positions=self.decoder.embed_positions,
                            embed_scale=self.embed_scale,
                            pad_id=self.pad_id,
                            dropout=self.dropout,
                            decoder_layers=decoder_layers_ls,
                            # raw parameter
                            decoder_layer=decoder_layers_ls[0],
                            num_layers=self.decoder_layers,
                            norm=decoder_norm)



        # initialize parameters
        if self.use_deepnorm:
            deepnorm_init_fn=partial(deepnorm_init,N=self.encoder_layers,M=self.decoder_layers)
            self.apply(deepnorm_init_fn)
        # else:
        #     self.apply(self.reset_paramaters)


    def build_moe_layers(self,layer_name="TransformerEncoderLayer",
                     num_layers=6,moe_experts = [8,1] * 3
                     ):
        assert layer_name in ["TransformerEncoderLayer","TransformerDecoderLayer"]
        layers_list=[]
        for i in range(num_layers):
            cur_experts = moe_experts[i]
            ffn_layer = build_moe_layer( self.d_model,
                                         self.dim_feedforward,
                                        num_experts=cur_experts,
                                        gate_type=self.gate_type,
                                        top_k=self.top_k,
                                        drop=0.,
                                        activation='relu') \
                if cur_experts > 1 else Mlp(
                                            self.d_model,
                                            self.dim_feedforward,
                                            drop=0.,
                                            activation='relu')

            layer_i = eval(layer_name)(
                use_deepnorm=self.use_deepnorm,
                d_model=self.d_model,
                nhead=self.nheads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                attn_dropout=0.,
                act_dropout=0.,
                activation='relu',
                normalize_before=False,
                bias_attr=[True] *2 if layer_name=="TransformerEncoderLayer" else [True] *3,
                )
            #
            layer_i.mlp = ffn_layer

            # need attention
            if layer_name == "TransformerDecoderLayer":
                layer_i.cross_attn.need_weights = self.need_attn

            layers_list.append(layer_i)

        return LayerList(layers_list)

    def get_balance_loss(self,alpha=1e-2):
        ''' gshard?: alpha: [1e-5,1e-1]
            we use an α = 10−2 which was sufficiently large to ensure load balancing
            while small enough to not to overwhelm the
            primary cross-entropy objective
         '''
        def layer_loss(layers):
            total_loss=0
            num=0
            for layer in layers:
                sub_layer = layer.mlp
                if isinstance(sub_layer, MoELayer):
                    num += 1
                    l = sub_layer.gate.get_loss()
                    if l is not None: total_loss += l
            return total_loss,num
        loss1,num1=layer_loss(self.encoder.layers)
        loss2,num2=layer_loss(self.decoder.layers)
        avg_moe_loss=(loss1+loss2)/(num1+num2) if (num1+num2)>0 else 0
        return alpha * avg_moe_loss


def _create_transformer(variant, is_test, pretrained_path, args):
    fleet.init(is_collective=True)
    model = MoETransformer(**args)
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
'''
moe配置:
1.gate?
2.数量：2、4、8？
3.前置还是后置：如41，14？
4.解码器需要吗： decoder： 6*1， 411111，或41111
5.层数：2层？3层？
6.容量
'''
# @register_model_arch("transformer_iwslt_de_en")
# def transformer_iwslt_de_en(is_test=False, pretrained_path=None, **kwargs):
#     for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
#     model_args = dict(encoder_layers=6,
#                       decoder_layers=6,
#                       d_model=512,
#                       nheads=4,
#                       dim_feedforward=1024,
#                       share_embed=True,
#                       share_all = True,
#                       use_deepnorm=False,
#                       **kwargs)
#     model_args = base_architecture(model_args)
#     model = _create_transformer('transformer_iwslt_de_en', is_test, pretrained_path, model_args)
#     return model
from ppseq.models import register_model_arch


@register_model_arch("transformer_de_en_naive")
def transformer_de_en_naive(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=4,
                      dim_feedforward=1024,
                      share_embed=True,
                      top_k=2,
                      enc_moe_experts=[4, 1] * 3,
                      dec_moe_experts=[4, 1] * 3,
                      use_deepnorm=False,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_de_en_naive', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_de_en_gshard")
def transformer_de_en_gshard(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=4,
                      dim_feedforward=1024,
                      share_embed=True,
                      # num_experts=8, # 改掉
                      gate_type="gshard",
                      top_k=2,
                      enc_moe_experts=[4, 1] * 3,
                      dec_moe_experts=[4, 1] * 3,
                      use_deepnorm=False,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_de_en_gshard', is_test, pretrained_path, model_args)
    return model

@register_model_arch("transformer_de_en_switch")
def transformer_de_en_switch(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers=6,
                      decoder_layers=6,
                      d_model=512,
                      nheads=4,
                      dim_feedforward=1024,
                      share_embed=True,
                      # num_experts=8, # 改掉
                      gate_type="switch",
                      top_k=1,
                      enc_moe_experts=[4, 1] * 3,
                      dec_moe_experts=[4, 1] * 3,
                      use_deepnorm=False,
                      **kwargs)
    model_args = base_architecture(model_args)
    model = _create_transformer('transformer_de_en_switch', is_test, pretrained_path, model_args)
    return model