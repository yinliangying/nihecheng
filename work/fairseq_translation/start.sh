#!/usr/bin/env bash
cd ../..

'
# Binarize the dataset:
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en

exit 0
'

# Train the model (better for a single GPU setup):
mkdir -p checkpoints/fconv
CUDA_VISIBLE_DEVICES=0
fairseq-train data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1  --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 200 \
  --arch fconv_iwslt_de_en --save-dir checkpoints/fconv

# Generate:
fairseq-generate data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/fconv/checkpoint_best.pt \
  --batch-size 128 --beam 5 --remove-bpe