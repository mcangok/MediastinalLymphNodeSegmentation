#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="30g"

docker volume create lnq2023-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --gpus all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v lnq2023-output-$VOLUME_SUFFIX:/output/ \
        lnq2023




docker run --rm -v lnq2023-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output:/expected_output/ \
        biocontainers/simpleitk:v1.0.1-3-deb-py3_cv1 python3 -c """
import SimpleITK as sitk
import os
import shutil

print('testing now')
print(os.listdir('/output'))
print(os.listdir('/output/images'))
print(os.listdir('/output/images/mediastinal-lymph-node-segmentation'))

print('expected_output folder')
print(os.listdir('/expected_output'))
print(os.listdir('/expected_output/images'))

shutil.copy2('/output/images/mediastinal-lymph-node-segmentation/inference_seg.nii.gz', '/expected_output')
shutil.copy2('/output/images/mediastinal-lymph-node-segmentation/inference_img_cropped.nii.gz', '/expected_output')


output = sitk.ReadImage('/output/images/mediastinal-lymph-node-segmentation/inference_seg.nii.gz')
"""

docker volume rm lnq2023-output-$VOLUME_SUFFIX
