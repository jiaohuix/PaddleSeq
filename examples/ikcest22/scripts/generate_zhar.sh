directions=("zh_ar" "ar_zh")
ckpts=("output/ckpt_zhar/epoch_final"
        "output/ckpt_arzh/epoch_final")

for ((i=0;i<${#directions[@]};i++))
  do  
      direct=${directions[$i]}
      ckpt=${ckpts[$i]}
      echo "------------------------------------------------------------generate ${direct}....------------------------------------------------------------"
      python paddleseq_cli/generate.py -c examples/ikcest22/configs/${direct}.yaml --pretrained $ckpt --only-src
      cat output/generate.txt | grep -P "^H" | sort -V | cut -f 3- > ${direct}.rst
  done

zip -r trans_result.zip *.rst

echo "all done"

