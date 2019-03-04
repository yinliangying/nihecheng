#!/usr/bin/env bash

__script_dir=$(cd `dirname $0`; pwd)
cd ${__script_dir}
SOURCE_DATA_DIR=../../input/competition/
USR_DIR=${__script_dir}/my_problem/
TMP_DIR=${__script_dir}/tmp/
PROBLEM=my_reaction_subword  #text2text_tmpdir_tokens
DATA_DIR=${__script_dir}/data_token/
MODEL=transformer_src_features
TRAIN_DIR=${__script_dir}/train_tiny_token/  #transformer_tiny
HPARAMS_SET=transformer_sfeats_hparams
DECODE_TO_FILE=${TMP_DIR}/result.txt
DECODE_FROM_FILE=${TMP_DIR}/test_sources
RESULT_FILE=${TMP_DIR}/result.json
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR


if [ $1 -eq 1 ]; then
    #1.pre-process-data
    python s1_token_process_cano_keku.py ${SOURCE_DATA_DIR}/train.txt ${TMP_DIR} train
    python s1_token_process_cano_keku.py ${SOURCE_DATA_DIR}/test.txt ${TMP_DIR} test
    python s1_build_token_vocab.py ${TMP_DIR}
    cp ${TMP_DIR}/vocab.token ${DATA_DIR}
    exit 0
elif [ $1 -eq 2 ];then
    #2. create t2t  data form
    t2t-datagen \
      --t2t_usr_dir=$USR_DIR \
      --data_dir=$DATA_DIR \
      --tmp_dir=$TMP_DIR \
      --problem=$PROBLEM

      exit 0
elif [ $1 -eq 3 ];then
    #3.train
    t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR\
  --train_steps 1000000\
  --hparams_set=$HPARAMS_SET  \
  #--hparams='batch_size=4096'\
  #--hparams='batch_size=4000,num_hidden_layers=4,hidden_size=16,filter_size=16,num_heads=4'

    exit 0
elif [ $1 -eq 4 ];then
    #4.decode
    t2t-decoder \
        --t2t_usr_dir=$USR_DIR  \
        --problem=$PROBLEM \
        --data_dir=$DATA_DIR \
        --model=${MODEL} \
        --output_dir=$TRAIN_DIR \
        --decode_from_file=${DECODE_FROM_FILE} \
        --decode_to_file=$DECODE_TO_FILE \
        --hparams_set=$HPARAMS_SET \
        --decode_hparams='alpha=0.6' \
        #--decode_hparams='return_beams=True,beam_size=10' # return topN(N=beam_size) result split by \t for each sample

    exit 0

elif [ $1 -eq 5 ];then
    #5.result
    #TEST_ID_FILE=${TMP_DIR}/test_id
    #create submission data
    #TEST_ID_LEN=`wc -l ${TEST_ID_FILE} | awk '{print $1}'`
    DECODE_TO_FILE_LEN=`wc -l ${DECODE_FROM_FILE}  | awk '{print $1}'`
    python submit.py $DECODE_TO_FILE  ${RESULT_FILE}
fi

#!/usr/bin/env bash