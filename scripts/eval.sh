echo "bash eval.sh <testfile> <genfile>"
testfile=$1
genfile=$2
sed "s/@@ //g" $testfile > ref
cat $genfile |  grep -P "^H" | sort -V | cut -f 3- > hypo
perl scripts/multi-bleu.perl ref < hypo
