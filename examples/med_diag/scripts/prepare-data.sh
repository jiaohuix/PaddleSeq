#!/bin/bash
if [ $# -lt 2 ];then
  echo "usage: bash $0 <infolder> <outfolder> "
  exit
fi


infolder=$1
outfolder=$2

if [ ! -d $outfolder ];then
  mkdir -p $outfolder
fi

echo "split dataset..."
cut -f2 -d"," $infolder/train.csv > train.src
cut -f3 -d"," $infolder/train.csv > train.tgt
cut -f2 -d"," $infolder/preliminary_a_test.csv > $outfolder/test.src

echo "train dev split..."
python nmt_data_tools/my_tools/train_dev_split.py src tgt train $outfolder  500
mv $outfolder/dev.src  $outfolder/valid.src
mv $outfolder/dev.tgt  $outfolder/valid.tgt


echo "build vocab..."
# -e开启转义
#echo -e "<s>\n<pad>\n</s>\n<unk>" > $outfolder/vocab.src
#for i in {9..1299}; do echo $i >> $outfolder/vocab.src; done

echo "<s>" > $outfolder/vocab.src
echo "<pad>" >> $outfolder/vocab.src
echo "</s>" >> $outfolder/vocab.src
echo "<unk>" >> $outfolder/vocab.src
for i in $(seq 9 1299)
do
  echo $i >> $outfolder/vocab.src
done

cp $outfolder/vocab.src $outfolder/vocab.tgt

ls $outfolder

echo "all done!"
