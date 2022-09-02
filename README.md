# PaddleSeq
## 快速开始

### 1.准备工作

```shell
# 克隆至本地
git clone https://github.com/MiuGod0126/STACL_Paddle.git
cd STACL_Paddle
# 安装依赖
pip install -r requirements
```

### 2.目录结构

```
├── ckpt # 权重
├── configs #配置
├── dataset # 数据
│   ├── ccmt21
│   ├── bstc
│   ├── enes21
├── decode # waitk结果文件夹
├── examples # 回译代码
├── models #模型文件
├── reader # 数据加载
├── paddleseq_cli 
│   ├── preprocess.py # 二值化
│   ├── train.py # 训练
│   ├── valid.py # 评估
│   ├── generate.py # 生成
│   ├── config.py # 命令行参数
├── scripts # 训练、微调、评估、waitk预测、平均权重脚本
├── tools # al评估
├── output # 输出文件
├── requirements.txt # 依赖
├── README.md
```

### 3.数据处理

#### 3.1 预处理

- 分词：对于中文先用jieba分词；然后分别对中英（西）使用moses的normalize-punctuation和tokenizer。（事实上中文不需要用moses，而moses在解码后需要de-tokenizing）。
- 长度过滤：对于中英，过滤掉长度1-250，并且长度比例超过1:2.5或2.5:1的平行语料；对于英西，过滤掉长度1-250，并且长度比例超过1:1.5或1.5:1的平行语料。
- 语言标识过滤(lang id)：使用fasttext过滤掉源端或目标端任意一边不匹配语言标识的平行文本。
- 对于中文的单语，进行了去重，减少了3m。
- truecase： 对英西两种语言使用truecase，自动判断句中名字、地点等，选择何时的大小写形式，而非直接使用小写，解码后需要de-truecaseing。(中文不用，且此步需要训练模型，处理非常耗时)。
- BPE(双字节编码)分子词： 对于中英，各自使用32K次操作；对于英西，共享32K的子词词表；其中中->英的词表包含ccmt、bstc的训练集，以及ccmt的单语中文语料。

#### 3.2 二进制

​	本项目支持两种格式的数据输入，一是文本对，二是fairseq的二进制数据（能压缩一半），以bstc为例，若要生成bin数据，命令如下(bin数据的使用见：[这](#bin_load))：

```shell
workers=1
TEXT=dataset/bstc
python paddleseq_cli/preprocess.py \
        --source-lang zh --target-lang en \
        --srcdict $TEXT/vocab.zh --tgtdict  $TEXT/vocab.en \
        --trainpref $TEXT/asr.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/dev.bpe  \
        --destdir data_bin/bstc_bin --thresholdtgt 0 --thresholdsrc 0 \
        --workers $workers
#⭐或
bash scripts/preprocess.sh
```

结果如下所示：

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

**注意：在windows上支持workers>1,而在aistudio上目前只能用workers=1**

### 4.模型训练

以提供的中英ccmt翻译数据为例，可以执行如下命令进行模型训练：

```shell
# 单卡或多卡训练（设置ngpus）
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
#⭐或
bash scripts/train_full.sh
# 模型验证
python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml  --pretrained ckpt/model_best_zhen --eval
```

对于中英在ccmt上训练后，还需用zhen_bstc.yaml进行微调：

```
├── configs #配置文件
│   ├── enes_un.yaml # 英西整句训练
│   ├── enes_waitk.yaml # 英西waitk
│   ├── zhen_ccmt.yaml # 中英整句预训练
│   ├── zhen_bstc.yaml # 中英整句微调
│   ├── zhen_bstc_bin.yaml # 中英整句微调(二进制)
│   ├── zhen_waitk.yaml # 中英waitk

```

除此之外，当数据量太大的时候有两种方法：

<a id="bin_load"></a>

1. 部分训练：修改配置文件中**train.train_data_size**，默认-1即加载全部。适用于需要快速加载调试，或用少量语料微调模型。
2. ⭐部分加载（全量训练）：使用迭代器，先获取一个pool大小的数据，然后再用MapDataset全量加载动态组batch，极大提升了数据加载速度并且防止爆内存。若要使用此功能，先使用数据准备中的命令生成二进制数据，然后修改配置文件中**data.use_binary**，**data.lazy_load**为True（别忘了修改数据前缀），详见**zhen_bstc_bin.yaml**，训练命令不变。



### 5.预测评估

以ccmt21为例，模型训练完成后可以执行以下命令对指定文件中的文本进行翻译，默认将结果输出到output/generate.txt：

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



###  6.回译

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



## 参考链接

[1. STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://www.aclweb.org/anthology/P19-1289.pdf)

[2.SimulTransBaseline](https://aistudio.baidu.com/aistudio/projectDetail/315680/)：

[3.PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/simultaneous_translation/stacl)

[4.fairseq](https://github.com/pytorch/fairseq)

[5.ConvS2S_Paddle](https://github.com/MiuGod0126/ConvS2S_Paddle)

[6.DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

