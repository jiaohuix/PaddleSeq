#!/bin/bash
echo "function: detokenize and detruecase."
if [ $# -lt 2 ];then
  echo "usage: bash $0 <prefix> <tgt>"
  exit
fi
prefix=$1
TRG=$2
DETOK=scripts/detokenizer.perl
DETRUE=scripts/detruecase.perl

# detokenize
cat $prefix.$TRG | perl $DETOK  -l $TRG > $prefix.detok.$TRG

# detruecase
perl $DETRUE  < $prefix.detok.$TRG > $prefix.detrue.$TRG

echo "write to $prefix.detrue.$TRG"
