#!/usr/bin/env bash

'
fairseq-preprocess \
  --trainpref names/train --validpref names/valid --testpref names/test \
  --source-lang input --target-lang label \
  --destdir names-bin --output-format raw
'


fairseq-train names-bin \
  --task simple_classification \
  --arch pytorch_tutorial_rnn \
  --optimizer adam --lr 0.001 --lr-shrink 0.5 \
  --user-dir my-module \
  --max-tokens 1000