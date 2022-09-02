#!/bin/bash

function func_shard(){
    workers=$1
    infile=$2

    lines=`cat $infile | wc -l`
    echo "total lines: $lines"

    shard_lines=$((lines/${workers})) # lines of each shard
    tail_lines=$((lines%${workers})) # lines of remain

    for i in $(seq 0 $(($workers-1)))
    do
      tail -n +$(($i*$shard_lines+1)) $infile | head -n $shard_lines > $infile.${i}
    done
    tail -n +$(($workers*$shard_lines+1))  $infile>> $infile.$(($workers-1))

    echo "--------------File ${inflie} has been divides into ${workers} shards.--------------"

}

function func_merge_shard(){
    workers=$1
    shard_prefix=$2
    outfile=$3

    for i in $(seq 0 $(($workers-1)))
    do
      cat $shard_prefix.${i} >> $outfile
      rm $shard_prefix.${i}
    done

      echo "---------------${workers} shards have been merged into ${outfile}.--------------"

}

function func_paral_process(){
    workers=$1
    py_script=$2
    infile=$3
    outfile=$4
    # 1.shard [infile->infile.idx]
    func_shard $workers $infile

    # 2.parallel process [infile.idx->infile.tmp.idx]
    for i in $(seq 0 $(($workers-1)))
    do
      (
      echo "----------------------processing shard: ${i}.----------------------"
      python $py_script $infile.${i} $infile.tmp.${i}
      rm $infile.${i}
      )&
    done
    wait

    # 3.merge [infile.tmp.idx->outfile]
    if [ -e $outfile ];then
      rm $outfile
    fi
    func_merge_shard $workers ${infile}.tmp $outfile


}