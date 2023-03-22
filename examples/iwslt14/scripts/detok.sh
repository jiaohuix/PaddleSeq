echo "src tgt infolder outfolder"
src=$1
tgt=$2
infolder=$3
outfolder=$4
#export TOOLS=$PWD/nmt_data_tools/
SCRIPTS=$TOOLS/mosesdecoder/scripts

if [ ! -d $outfolder ];then
  mkdir -p $outfolder
fi

for prefix in train valid test
  do
    echo "process $prefix ..."
    sed "s/@@ //g"  $infolder/$prefix.$src | perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q > $outfolder/$prefix.$src
    sed "s/@@ //g"  $infolder/$prefix.$tgt | perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q > $outfolder/$prefix.$tgt
  done


#cat $FILE | grep -P "^D" | sort -V | cut -f 3- > $FILE.tok
#sed -r 's/(@@ )|(@@ ?$)//g' $REF > $REF.tok

