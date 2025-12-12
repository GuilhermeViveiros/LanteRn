# bash scripts/eval.sh lambda_mse_0.2
# bash scripts/eval.sh lambda_mse_latent_size_4_lambda_0.1
# bash scripts/eval.sh lambda_mse_latent_size_8_lambda_0.1
# bash scripts/eval.sh sft_mse_lt_4_lambda_0.0
# bash scripts/eval.sh sft_mse_lt_4_lambda_0.1
# bash scripts/eval.sh sft_mse_lt_4_lambda_0.2


CHECKPOINTS=(200 400 600 800 900)
MODELS_FOLDER="/mnt/scratch-artemis/gviveiros/lantern/checkpoints"
MODEL_NAME="${1}"

# check if the model name is valid
if [ -z "$MODEL_NAME" ]; then
    echo "Model name is required"
    exit 1
fi

OUTPUT_DIR="${MODEL_NAME}"
MODEL_FOLDER="${MODELS_FOLDER}/${MODEL_NAME}"
echo "Evaluating models in $MODEL_FOLDER folder"
echo "Output directory: $OUTPUT_DIR"

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "Evaluating checkpoint '$MODEL_FOLDER/checkpoint-${ckpt}' and saving results to $OUTPUT_DIR"
    sbatch --job-name="lantern_${MODEL_NAME}_${ckpt}" scripts/eval_viscot.sh "$MODEL_FOLDER/checkpoint-${ckpt}" "$OUTPUT_DIR" 
done