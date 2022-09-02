#!/bin/bash
source scripts/common_func.sh


# params
if [ $# -ne 2 ];then
  echo "usage: bash $0 [workers] [infile] "
  exit
fi

workers=$1
infile=$2

func_shard $workers $infile