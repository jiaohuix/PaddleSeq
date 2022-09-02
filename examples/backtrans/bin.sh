# only windows can use workers>1
python examples/backtransextract_bt_data.py --minlen 1 --maxlen 250 --ratio 2.5 --output output/ccmt_bt --srclang zh --tgtlang en  --min-lprob -3 output/bt_ft_ccmt/bt/generate_enzh.txt

