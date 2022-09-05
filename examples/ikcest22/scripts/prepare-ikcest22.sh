# params
centric_lang=zh
noncentric_langs=("th" "fr" "ru")
raw_folder=datasets/raw
tmp_folder=datasets/tmp
bpe_folder=datasets/bpe
valid_len=1000
bpe_ops=18000
URL="https://dataset-bj.cdn.bcebos.com/qianyan/datasets.tar.gz"

function build_folder() {
    local folder=$1
    if [ ! -d $folder ];then
        mkdir -p $folder
    fi
}

# dowanload datasets
#if [ ! -d datasets ]; then
#  wget $URL
#  tar -xvf datasets.tar.gz
#fi
if [ ! -e datasets.tar.gz ];then
    wget $URL
fi
tar -xvf datasets.tar.gz

build_folder $raw_folder
build_folder $tmp_folder
build_folder $bpe_folder

# extract raw data
for lang in ${noncentric_langs[@]}
  do
      lang_folder=$raw_folder/${centric_lang}_${lang}
      build_folder $lang_folder
      # cut train
      cut -f 1 datasets/${centric_lang}_${lang}.train > $lang_folder/train.ikcest.${centric_lang}
      cut -f 2 datasets/${centric_lang}_${lang}.train > $lang_folder/train.ikcest.${lang}
      rm datasets/${centric_lang}_${lang}.train && rm datasets/${lang}_${centric_lang}.train
      # train dev split
      python nmt_data_tools/my_tools/train_dev_split.py $centric_lang $lang  $lang_folder/train.ikcest $lang_folder/ $valid_len
      mv $lang_folder/dev.$centric_lang $lang_folder/valid.$centric_lang
      mv $lang_folder/dev.$lang $lang_folder/valid.$lang
      # rename test
      mv datasets/${centric_lang}_${lang}.test $lang_folder/test.${centric_lang}_${lang}.${centric_lang}
      mv datasets/${lang}_${centric_lang}.test  $lang_folder/test.${lang}_${centric_lang}.${lang}
  done



function tokenize() {
    if [ $# -lt 4 ];then
      echo "usage:  $0 <infolder> <outfolder> <prefix> <lang>"
      exit
    fi
    # params, local var prevent changing global var
    local infolder=$1
    local outfolder=$2
    local prefix=$3
    local lang=$4
    workers=4
    echo "tokenize $infolder/$prefix.$lang ..."
    cut_langs=("zh","th") # language need cut words
    if echo ${cut_langs[@]} | grep -w ${lang} &>/dev/null
      then
          python nmt_data_tools/my_tools/cut_multi.py $infolder/$prefix.$lang   $outfolder/$prefix.tok.$lang $workers $lang
      else
        cp $infolder/$prefix.$lang   $outfolder/$prefix.tok.$lang
        echo "language $lang dont's need cut."

    fi
}


function learn_apply_bpe() {
    if [ $# -lt 3 ];then
      echo "usage:  $0 <folder> <prefix> <lang>"
      exit
    fi
    local folder=$1
    local prefix=$2
    local lang=$3
    tok_file=$folder/$prefix.tok.$lang
    bpe_file=$folder/$prefix.bpe.$lang
    if [ -e $folder/code.$lang ]
        then
            echo "----apply bpe to $tok_file----"
            subword-nmt apply-bpe -c $folder/code.$lang < $tok_file  >  $bpe_file
        elif [ "$prefix"x == "train"x ]
        then
            echo "----learn bpe code, and apply to $tok_file----"
            subword-nmt learn-bpe -s $bpe_ops < $tok_file  > $folder/code.$lang
            subword-nmt apply-bpe -c  $folder/code.$lang < $tok_file >  $bpe_file
        else
          echo "no training data error."
      fi
}


# tokenize and bpe
for folder in `ls $raw_folder`
  do
    # build tmp,bpe subfolder
    build_folder $tmp_folder/$folder
    build_folder $bpe_folder/$folder
    langs_pair=(${folder//_/ })
    tgt_lang=${langs_pair[1]}

    for lang in ${langs_pair[@]}
      do
          for prefix in train valid test.${centric_lang}_${tgt_lang} test.${tgt_lang}_${centric_lang}
            do
                if  [ -e $raw_folder/$folder/$prefix.$lang ]; then
                    # 1.tokenize
                    tokenize  $raw_folder/$folder/  $tmp_folder/$folder $prefix $lang
                    # 2.learn and apply bpe
                    learn_apply_bpe $tmp_folder/$folder $prefix $lang
                    # 3.build vocab
                    if  [ "$prefix"x == "train"x ]; then
                        python nmt_data_tools/my_tools/build_dictionary.py $tmp_folder/$folder/$prefix.bpe.$lang
                        python nmt_data_tools/my_tools/json2vocab.py $tmp_folder/$folder/$prefix.bpe.$lang.json $bpe_folder/$folder/vocab.$lang
                    fi
                    # 4.copy train,valid, bpe code to bpe_folder
                    cp $tmp_folder/$folder/$prefix.bpe.$lang $bpe_folder/$folder/$prefix.$lang
                    cp $tmp_folder/$folder/code.$lang $bpe_folder/$folder/
                fi
            done

      done
  done


echo "all done!"



