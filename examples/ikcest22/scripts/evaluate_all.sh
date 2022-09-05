directions=("zh_th" "th_zh" "zh_fr" "fr_zh" "zh_ru" "ru_zh")
data_paths=("zh_th" "zh_th" "zh_fr" "zh_fr" "zh_ru" "zh_ru")
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
      python paddleseq_cli/generate.py -c examples/ikcest22/configs/${direct}.yaml --pretrained $ckpt --test-pref datasets/bpe/${data_paths[$i]}/valid
  done

echo "all done"

