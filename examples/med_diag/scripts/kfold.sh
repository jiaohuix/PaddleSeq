#!/bin/bash
if [ $# -lt 4 ];then
  echo "usage: bash $0 <src> <tgt> <prefix> <outfolder> <kfold=5>(opt) <k=0>(opt) <seed=1>(opt)"
  echo "note: <k> = [0...kfold-1]"
  exit
fi
src=$1
tgt=$2
prefix=$3
outfolder=$4
kfold=${5:-5}
k=${6:0}
seed=${7:1}
k_padded=$(printf "%02d" $k) # 圆括号！

srcfile=$prefix.$src
tgtfile=$prefix.$tgt

if [ ! -d $outfolder ];then
  mkdir -p $outfolder
fi
tmpfile=$outfolder/tmp/
mkdir -p $tmpfile

echo "shuffle..."
paste $srcfile $tgtfile > $tmpfile/merged_dataset
shuf --random-source=<(yes $seed) $tmpfile/merged_dataset >  $tmpfile/shuffled_dataset

echo "split to kfold..."
# 划分成k份
split -d -n l/$kfold  $tmpfile/shuffled_dataset $tmpfile/dataset_

# valid
echo "build valid..."
cut -f1 $tmpfile/dataset_$k_padded > $outfolder/valid.$src
cut -f2 $tmpfile/dataset_$k_padded > $outfolder/valid.$tgt

# train
#cat dataset_{0..$((i-1))} dataset_$((i+1))-$(($K-1))
echo "build train..."
for ((i=0;i<$kfold;i++))
do
    i_padded=$(printf "%02d" $i)
    # 跳过第k折作为训练集
    if [ "${i_padded}"x == "${k_padded}"x ];then
      continue
    fi
    cat dataset_${i_padded} > test.txt
    cut -f1 $tmpfile/dataset_$i_padded >> $outfolder/train.$src
    cut -f2 $tmpfile/dataset_$i_padded >> $outfolder/train.$tgt
done

rm -rf
echo "outfolder: [$outfolder]"
ls $outfolder
echo "all done!"
