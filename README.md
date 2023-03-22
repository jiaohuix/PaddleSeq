# PaddleSeq

For Machine Translation using PaddlePaddle.

## Update:

23/3/14:  add 2 criterions:  rdrop、simcut，change the 'criterion ' in the configuration file to take effect. (development branch)



## Requirements

```shell
# install ppseq
git clone https://github.com/MiuGod0126/PaddleSeq.git
cd PaddleSeq
pip install -r requirements.txt && pip install -e .
# data tools
git clone https://github.com/MiuGod0126/nmt_data_tools.git
pip install -r nmt_data_tools/requirements.txt
```



## Examples

1. [IWSLT14 DE EN](./examples/iwslt14)  (development branch)
2. ⭐[IKCEST22](examples/ikcest22/README.md)



使用hydra管理参数：

```shell
# 1.环境变量
export CONFIG=$PWD/configs/nmt.yaml
# 2.修改yaml中的路径为绝对路径
sed -i "s|datasets|$PWD/datasets|g"  configs/nmt.yaml
# 3.启动hydra训练
hydra_train key=value...
```



## REF

[1.PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation)

[2.fairseq](https://github.com/pytorch/fairseq)

[3.ConvS2S_Paddle](https://github.com/MiuGod0126/ConvS2S_Paddle)

[4.STACL_Paddle](https://github.com/MiuGod0126/STACL_Paddle)

