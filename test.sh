#!/bin/bash

gpus=1
base_dir="/root/logdir/checkpoints/samples"
last_dir=$(ls -td "$base_dir"/*/ | head -1)

# Initialize degree value
degree=0.8

# Loop through the process with decreasing degree values
for i in {1..4}; do
    echo "Running with $degree"
    # Use $degree in the command to reference its value
    torchrun --standalone --nproc_per_node=$gpus scripts/sample_diffusion_cvd_multi_gpu.py \
        -r /root/CVD-Diffusion/ldm/logs/checkpoints/last.ckpt \
        -l /root/logdir \
        --batch_size 8 \
        -c 50 \
        -e 0.0 \
        --cvd_degree $degree \
        --gs 70 \
        --seed=0-479 \
        --cvd_alpha=0.3 \
        --backward_iters=10 \
        --init_latent_load_offset=0

    last_sub_dir=$(ls -td "$last_dir"/*/ | head -1)
    python utils/format_image_grid.py --images_path="$last_sub_dir/img" --outdir="/root/expout$degree"

    # Decrease degree for the next iteration
    degree=$(awk "BEGIN {print $degree - 0.2}")
done

echo "All done."
