#!/bin/bash

MODEL_PATH="/Users/anu/PycharmProjects/Siglip/gemma-keras-gemma_1.1_instruct_2b_en-v3"
PROMPT="this building is "
IMAGE_FILE_PATH="/Users/anu/PycharmProjects/Siglip/P.jpeg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="True"

python3 inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \