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

# detruecase
perl $DETRUE  < $prefix.$TRG > $prefix.detrue.$TRG
echo "write to $prefix.detrue.$TRG"

# detokenize
cat  $prefix.detrue.$TRG | perl $DETOK  -l $TRG > $prefix.detok.$TRG
echo "write to $prefix.detok.$TRG"

echo "all done!"