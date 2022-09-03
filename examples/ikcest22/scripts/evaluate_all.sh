directions=("zh_th" "th_zh" "zh_fr" "fr_zh" "zh_ru" "ru_zh")
ckpts=("output/ckpt_zhth/epoch_final"
        "output/ckpt_thzh/epoch_final"
        "output/ckpt_zhfr/epoch_final"
        "output/ckpt_frzh/epoch_final"
        "output/ckpt_zhru/epoch_final"
        "output/ckpt_ruzh/epoch_final")

for ((i=0;i<${#directions[@]};i++))
  do  
      direct=${directions[$i]}
      ckpt=${ckpts[$i]}
      echo "------------------------------------------------------------evaluate ${direct}....------------------------------------------------------------"
      python paddleseq_cli/generate.py -c examples/ikcest22/configs/${direct}.yaml --pretrained $ckpt --test-pref datasets/bpe/${direct}/valid
  done

echo "all done"
