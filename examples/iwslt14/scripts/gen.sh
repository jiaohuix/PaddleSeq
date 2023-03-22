fairseq-generate $DATA --path $CKPT  --gen-subset test --beam 5 --lenpen 1 --max-tokens 8192 --remove-bpe

fairseq-generate  data-bin/iwslt14.tokenized.de-en/ --task translation --path checkpoint/iwslt14_de_en_bi/checkpoint_best.pt  --gen-subset test --beam 5 --lenpen 1 --max-tokens 8192 --remove-bpe > generate.bi.txt
# 原始多少？


# in-context
#--prefix-size  -1     # <0会不取，取最大吧
fairseq-generate  data-bin/iwslt14.tokenized.sim.de-en --task translation --path checkpoint/iwslt14_de_en_bi/checkpoint_best.pt  --gen-subset test --beam 5 --lenpen 1 --max-tokens 8192 --remove-bpe  --prefix-size  1024 --left-pad-target  > generate.sim.txt
# 后处理和评估

#assert (first_beam == target_prefix).all() 要注释掉， prefix_size = 1024
#cat $FILE | grep -P "^D" | sort -V | cut -f 3- > $FILE.tok
#sed -r 's/(@@ )|(@@ ?$)//g' $REF > $REF.tok

fairseq-generate  data-bin/iwslt14.tokenized.de-en --task translation --path  iwslt14_deen_bi/checkpoint_best.pt  --gen-subset test --beam 5 --lenpen 1 --max-tokens 8192 --remove-bpe  --prefix-size  1024 --left-pad-target  > generate.sim.txt