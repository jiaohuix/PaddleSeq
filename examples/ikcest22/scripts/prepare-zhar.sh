src=zh
tgt=ar
bpe_ops=10000
valid_num=1000
infolder=$1
outfolder=datasets/bpe/${src}_${tgt}
if [ ! -d $outfolder ];then
    mkdir -p $outfolder
fi

# train dev split
cp $infolder/train.$src $infolder/train.tmp.$src && cp $infolder/train.$tgt $infolder/train.tmp.$tgt
python nmt_data_tools/my_tools/train_dev_split.py $src $tgt $infolder/train.tmp $infolder/ $valid_num # train.lang/ dev.lang
mv $infolder/dev.$src  $infolder/valid.$src && mv $infolder/dev.$tgt  $infolder/valid.$tgt


# tokenize
for prefix in train valid test.${src}_${tgt}
    do  
        python nmt_data_tools/my_tools/cut_multi.py  $infolder/$prefix.$src  $infolder/$prefix.tok.$src 4 zh
    done

# learn bpe
subword-nmt learn-bpe -s $bpe_ops < $infolder/train.tok.$src > $outfolder/code.$src
subword-nmt learn-bpe -s $bpe_ops < $infolder/train.$tgt > $outfolder/code.$tgt

# apply bpe
for prefix in train valid test.${src}_${tgt} test.${tgt}_${src}
    do  
        subword-nmt apply-bpe -c $outfolder/code.$src < $infolder/$prefix.tok.$src >  $outfolder/$prefix.$src
        subword-nmt apply-bpe -c $outfolder/code.$tgt < $infolder/$prefix.$tgt >  $outfolder/$prefix.$tgt 
    done

# vocab
python nmt_data_tools/my_tools/build_dictionary.py  $outfolder/train.$src
python nmt_data_tools/my_tools/build_dictionary.py  $outfolder/train.$tgt

# build paddle vocab
python nmt_data_tools/my_tools/json2vocab.py $outfolder/train.$src.json $outfolder/vocab.$src
python nmt_data_tools/my_tools/json2vocab.py $outfolder/train.$tgt.json $outfolder/vocab.$tgt

echo "all done!"
