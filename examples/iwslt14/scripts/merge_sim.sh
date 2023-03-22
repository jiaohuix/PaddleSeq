# add similar top1 to test
# 数据放LASER 目录下： bash merge.sh de en test.sim.top1 iwslt14.tokenized.de-en/ iwslt14_sim
echo "src tgt sim_prefix infolder outfolder"
src=$1
tgt=$2
sim_prefix=$3
infolder=$4
outfolder=$5

if [ ! -d $outfolder ];then
  mkdir -p $outfolder
fi

cp -r  $infolder/* $outfolder

# concat sim.src test.src
paste $sim_prefix.$src  $infolder/test.$src > $outfolder/test.$src
cp    $sim_prefix.$tgt  $outfolder/test.$tgt


# preprocess
# 需要用提前做好的词典，防止顺序错误
TEXT=$outfolder
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.sim.de-en \
    --srcdict $infolder/dict.$src.txt --tgtdict $infolder/dict.$tgt.txt --workers 20
#    --joined-dictionary --workers 20
