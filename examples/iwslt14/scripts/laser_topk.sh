echo "src tgt folder"
src=$1
tgt=$2
folder=$3


# 合并train和valid
cat $folder/train.$src $folder/valid.$src > $folder/database.$src
cat $folder/train.$tgt $folder/valid.$tgt > $folder/database.$tgt
cp  $folder/test.$src  $folder/query.$src
cp  $folder/test.$tgt  $folder/query.$tgt

# 编码
bash tasks/embed/embed.sh $folder/database.$tgt  $folder/database.$tgt.bin
bash tasks/embed/embed.sh $folder/query.$tgt     $folder/query.$tgt.bin

# 对目标语言搜索
python ../search/laser/laser_search.py -d  $folder/database.$tgt.bin -q  $folder/query.$tgt.bin -o sim  -k 2 -b 512  --index IVF --nlist 100

cut -f1 sim.idx > sim.top1.idx
# 需要把train和dev的tgt拼接起来
#cat iwslt14.tokenized.de-en/train.en iwslt14.tokenized.de-en/valid.en  > db.en
#cat iwslt14.tokenized.de-en/train.de iwslt14.tokenized.de-en/valid.de  > db.de

#python ../search/laser/extract_text_by_idx.py db.en sim.top1.idx   test.sim.top1.en
#python ../search/laser/extract_text_by_idx.py db.de sim.top1.idx   test.sim.top1.de
