directions=("zh_ar" "ar_zh")
data_paths=("zh_ar" "ar_zh")
ckpts=("output/ckpt_zhar/epoch_final"
        "output/ckpt_arzh/epoch_final")

for ((i=0;i<${#directions[@]};i++))
  do
      direct=${directions[$i]}
      ckpt=${ckpts[$i]}
      echo "------------------------------------------------------------evaluate ${direct}....------------------------------------------------------------"
      python paddleseq_cli/generate.py -c examples/ikcest22/configs/${direct}.yaml --pretrained $ckpt --test-pref datasets/bpe/${data_paths[$i]}/valid
  done

echo "all done"

