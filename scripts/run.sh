ppseq_train -c configs/de_en.yaml --amp --max-epoch 40
ppseq_train -c output/ckpt_deen/epoch_40/model.yaml --amp --max-epoch 50 --eval-beam
ppseq_generate -c output/ckpt_deen/epoch_final/model.yaml
