# PaddleSeq
## Requirements

```shell
git clone https://github.com/MiuGod0126/PaddleSeq.git
cd PaddleSeq
pip install -r requirements.txt
git clone https://github.com/MiuGod0126/nmt_data_tools.git
pip install -r nmt_data_tools/requirements.txt
```



## Examples

1. IWSLT14 DE EN
2. [IKCEST22](examples/ikcest22/README.md)













**2.Directory Structure**

**3.Binarize (optional)**

```shell
workers=1
TEXT=data_path
SRC=zh
TGT=en
python paddleseq_cli/preprocess.py \
        --source-lang $SRC --target-lang $TGT \
        --srcdict $TEXT/vocab.$SRC --tgtdict  $TEXT/vocab$TGT \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test  \
        --destdir data-bin --thresholdtgt 0 --thresholdsrc 0 \
        --workers $workers
# or
bash scripts/preprocess.sh
```

result:

```
data_bin/bstc_bin/
    preprocess.log
    test.zh-en.en.idx
    test.zh-en.en.bin
    test.zh-en.zh.bin
    train.zh-en.en.bin
    test.zh-en.zh.idx
    train.zh-en.zh.bin
    train.zh-en.en.idx
    train.zh-en.zh.idx
    valid.zh-en.en.bin
    valid.zh-en.en.idx
    valid.zh-en.zh.bin
    valid.zh-en.zh.idx
```

**Note**: Workers>1 is supported on Windows, but currently only workers=1 can be used on aisudio

**4.Training Scripts**

```shell
ngpus=4
python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml \
                         --amp \
                         --ngpus $ngpus  \
                         --update-freq 4 \
                         --max-epoch 10 \
                         --save-epoch 1 \
                         --save-dir /root/paddlejob/workspace/output \
                         --log-steps 100 \
                         --max-tokens 4096 \

# 模型验证
python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml  --pretrained ckpt/model_best_zhen --eval
```

除此之外，当数据量太大的时候有两种方法：

<a id="bin_load"></a>

1. 部分训练：修改配置文件中**train.train_data_size**，默认-1即加载全部。适用于需要快速加载调试，或用少量语料微调模型。
2. ⭐部分加载（全量训练）：使用迭代器，先获取一个pool大小的数据，然后再用MapDataset全量加载动态组batch，极大提升了数据加载速度并且防止爆内存。若要使用此功能，先使用数据准备中的命令生成二进制数据，然后修改配置文件中**data.use_binary**，**data.lazy_load**为True（别忘了修改数据前缀），详见**zhen_bstc_bin.yaml**，训练命令不变。



**5.Generation Scripts**

```shell
python  paddleseq_cli/generate.py --cfg configs/zhen_ccmt.yaml \
				   --pretrained ckpt/model_best_zhen \
				   --beam-size 5 \
				   --generate-path generate.txt \
				   --sorted-path result.txt
				   # --only-src # 若test无目标文件用此参数

#⭐或
bash scripts/generate_full.sh
```

训练、验证曲线使用visualdl生成，命令为：

```shell
visualdl --logdir output/vislogs/zhen --port 8080
# 打开链接：localhost:8080
```



**6.Backtranslation**

1. (X,Y)训练前向模型F

   ```shell
   python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml --src-lang zh --tgt-lang en 
   ```

2. (Y,X)训练反向模型B

   ```shell
   python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml --src-lang en --tgt-lang zh
   ```

3. 平均权重

   ```shell
   # 如output/ckpt下有model_best_27.5 model_best_26.5 model_best_26.4三个文件
   # 默认取最好的k个
   k=3
   python scripts/average_checkpoints.py \
   	--inputs output/ckpt  \
   	--output output/ckpt/avg${k} --num-ckpts $k
   ```

4. 单语Y1分片（当数据太大时，分不同机器预测）

   ```shell
   workers=2
   infile= dataset/mono.en # 目标单语路径
   bash examples/backtrans/shard.sh $workers $infile
   ```

5. 模型B预测X1

   ```shell
   ckpt_dir=model_best_enzh
   mono_file=dataset/mono.en
   python paddleseq_cli/generate.py --cfg configs/zhen_ccmt.yaml \
   			--src-lang en --tgt-lang zh \
               --test-pref $mono_file --only-src \
               --pretrained  $ckpt_dir  --remain-bpe
   # 注意保留bpe结果，以便用于训练
   ```

6. 查看预测结果logprob分布:

   受反向模型B质量的影响，生成结果可能较差，体现在generate.txt中lprob分数较低，可以使用如下命令查看lprob分布（可用于在7抽取时设置过滤阈值min-lprob）：

   ```shell
   python examples/backtrans/lprob_analysis.py output/generate.txt
   ```

   结果如：

   ```
               lprobs
   count  4606.000000
   mean     -1.060325
   std       0.256854
   min      -2.578100
   25%      -1.225675
   50%      -1.054400
   75%      -0.890825
   max      -0.209400
   ```

7. 抽取平行语料P' (X1,Y1)

   ```shell
   python examples/backtrans/extract_bt_data.py \
   		--minlen 1 --maxlen 250 --ratio 2.5 --min-lprob -3 \
   		--output output/ccmt_ft --srclang zh --tgtlang en  \
   		 output/bt/generate*.txt
   # min-lprob可以过滤掉lprob低于该值的预测
   # --output是过滤出的前缀，原语向为zh-en,对于回译而言单语是en，生成的是zh；若是自训练（即前向模型F预测zh单语），需要改为--srclang en --tgtlang zj
   # output/bt/generate*.txt 是多个分片预测的结果，如generate1.txt、generate2.txt...
   ```

8. 合并(X,Y) (X1,Y1)并继续训练F,略...



## REF

[1. STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://www.aclweb.org/anthology/P19-1289.pdf)

[2.SimulTransBaseline](https://aistudio.baidu.com/aistudio/projectDetail/315680/)：

[3.PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/simultaneous_translation/stacl)

[4.fairseq](https://github.com/pytorch/fairseq)

[5.ConvS2S_Paddle](https://github.com/MiuGod0126/ConvS2S_Paddle)

[6.DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

