## IWSLT14 DE-EN

## 0 安装

```shell
python setup.py install 
```

## 1 训练

```shell
ppseq_train -c examples/iwslt14/configs/de_en.yaml
# 恢复训练，如10epoch
ppseq_train -c output/ckpt_deen/epoch_10/model.yaml
```

## 2 推理

```shell
#  最佳权重目录：output/ckpt_deen/model_best_31
ppseq_generate -c output/ckpt_deen/model_best_31/model.yaml
```

## 3 评估

```shell
# multi-bleu
bash scripts/eval.sh datasets/iwslt14/test.en output/generate.txt
```



| framework | model                     | tok bleu(test) | epoch |
| --------- | ------------------------- | -------------- | ----- |
| fairseq   | transformer_iwslt14_de_en | 34.66          | 41    |
| paddle    | transformer_iwslt14_de_en | 34.46          | 26    |

2023/2/13

