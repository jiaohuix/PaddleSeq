#!/bin/bash
if [ $# -lt 1 ];then
  echo "usage: bash $0 <yaml>"
  exit
fi
yaml=$1
ppseq_generate -c $yaml --only-src
for i in {0..2999}; do echo $i >> no; done
cat output/generate.txt  |  grep -P "^H" | sort -V | cut -f 3- > hypo
paste -d"," no hypo > submit.csv
echo "write to submit.csv succes!"
