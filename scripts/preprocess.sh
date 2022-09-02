# only windows can use workers>1
echo "bash preprocess.sh folder src tgt workers"
workers=$4
TEXT=$1
SRC=$2
TGT=$3
python paddleseq_cli/preprocess.py \
        --source-lang $SRC --target-lang $TGT \
        --srcdict $TEXT/vocab.$SRC --tgtdict  $TEXT/vocab.$TGT \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/valid  \
        --destdir data-bin/${SRC}_${TGT} --thresholdtgt 0 --thresholdsrc 0 \
        --workers $workers
