epochs=50
freq=1 # update frequence

directions=("zh_th" "th_zh" "zh_fr" "fr_zh" "zh_ru" "ru_zh")
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
for direct in ${directions[@]}
  do
      echo "------------------------------------------------------------training ${direct}....------------------------------------------------------------"
      ppseq_train -c examples/ikcest22/configs/${direct}.yaml   --update-freq $freq --max-epoch $epochs
  done


echo "all done"

