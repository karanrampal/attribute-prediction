#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Navigate to proj directory
cd "$(dirname "$0")"
cd ../

# Set variables
proj="smle-attribution-d237"
region="europe-west4"

job_name="fit_$(date +%Y%m%d_%H%M%S)"
image_uri="gcr.io/cloud-ml-public/training/pytorch-gpu.1-11"
job_dir="gs://attribute-models-bucket/fit-model"
package="${job_dir}/AttributePrediction-0.0.1-py3-none-any.whl"
tier="basic"
module_name="trainer.train"


echo "Building package"
make build
gsutil cp ./dist/*.whl ${job_dir}

echo "Submitting AI Platform PyTorch job"
gcloud ai-platform jobs submit training ${job_name} \
    --region ${region} \
    --master-image-uri ${image_uri} \
    --scale-tier ${tier} \
    --module-name ${module_name} \
    --packages ${package} \
    -- \
    --num_epochs 2 \
    --batch_size 64 \
    --learning_rate 0.001

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${job_name}