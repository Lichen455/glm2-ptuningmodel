PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python web_demo2.py \
    --model_name_or_path ../model/chatglm2-6b \
    --ptuning_checkpoint output11/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-2000 \
    --pre_seq_len $PRE_SEQ_LEN

