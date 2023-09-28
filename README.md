# MediastinalLymphNodeSegmentation Algorithm

Challenge Submission of the CompAI team for the MICCAI Lymph Node Quantification Challenge

# Challenge Setting

Accurate lymph node size estimation is critical for staging cancer patients, initial therapeutic management, and in longitudinal scans, assessing response to therapy. Current standard practice for quantifying lymph node size is based on a variety of criteria that use unidirectional or bidirectional measurements on just one or a few nodes, typically on just one axial slice. But humans have hundreds of lymph nodes, any number of which may be enlarged to various degrees due to disease or immune response. While a normal lymph node may be approximately 5 mm in diameter, a diseased lymph node may be several cm in diameter. The mediastinum, the anatomical area between the lungs and around the heart, may contain ten or more lymph nodes, often with three or more enlarged greater than 1 cm. Accurate segmentation in 3D would provide more information to evaluate lymph node disease. Full 3D segmentation of all abnormal lymph nodes in a scan promises to provide better sensitivity to detect volumetric changes indicating response to therapy. 

<br>

The LNQ2023 Challenge training dataset will consist of a unique set of high-quality pixel-level annotations of one or more clinically relevant lymph nodes in a dataset part of cancer clinical trials. The goal will be to segment all lymph nodes larger than 1 cm in diameter in the mediastinum. Participants will be provided with a subset of cases that are partially annotated (i.e. one node out of five), and evaluation of the algorithms will be performed on a distinct dataset that is fully annotated (i.e. all clinically relevant nodes).



## Using the model via the docker container

this GitHub repository already includes the model weights, which were used for the final submission for the MICCAI LNQ2023 challenge (https://github.com/StefanFischer/MediastinalLymphNodeSegmentation/tree/main/lnq_segmentation/nnunet/nnUNet_results). If you want to use the model you can use the test.sh script that will build a docker container and run inference on the image at location './test/images/mediastinal-ct/*.nii.gz' (only one image per run).

## Using the model via Grand Challenge Website

You can also use the algorithms docker container hosted at https://grand-challenge.org/algorithms/mediastinallymphnodesegmentation/



# Training of own Model
If you want to train a model on your own data, you just need to install and use nnUNetv2.

## Installation of nnUNetv2

Please follow the installation guideline of the official GitHub page of nnUNetv2: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md  <br>
Set the environment variable such that the 'nnUNet_raw'-environment variable links to the './lnq_segmentation/nnunet/nnUNet_raw/' path of your git repository, same for the other env vars from nnUNetv2.

## Get the ROI-cropped data (CT imgs + Labels)

The training data was preprocessed with the totalsegmentator toolbox v1: https://github.com/wasserth/TotalSegmentator  <br>
In the process.py script the code for preprocessing can be found if you want to perform it by yourself using the (masks_to_boxes, totalsegmentator, combine_masks)-functions.  <br>
Otherwise you can have all the data by contacting me.

## Prepare the data

Put images and labels into the './lnq_segmentation/nnunet/nnUNet_raw/Dataset100_model_training'-folder.

## Overwrite nnUNetv2 code

Now change the standard nnUNet code by overwriting the source code in your python env with the nnUNet code of this repo. For that just copy the content of './lnq_segmentation/nnunet/nnunet_code_changes/nnunetv2/' to the nnUNet pip/conda env installation location (in my conda env: '/home/*/anaconda3/envs/nnunet_raw/lib/python3.10/site-packages/nnunetv2'. This will change the preprocessing and set the correct training parameters.


## Train the model

Training your own model with the same datasets we used for the challenge will take roughly 33 hours on a NVIDIA A6000.

Preprocess data with nnUNetv2 (plans are already in nnUNet_preprocessed):  <br>

`
nnUNetv2_preprocess -d 100 -c 3d_fullres -np 16
`

<br>

`
nnUNetv2_train 100 3d_fullres all
`


## Inference with model

This will not include preprocessing (lung cropping via totalsegmentator) and postprocessing (pathologic lymph nodes are filtered by Shortest-Axis-Diameter >= 10mm), for that use or check the Docker container and the process.py-Script.

After training you can use the model as a standard nnUNetv2 model:  <br>

`
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 100 -c 3d_fullres -f all
`

In my docker submission i reduced the step size of the patch-based inference, such that there is a bigger ensemble effect. I think that leads to a more stable output, but will increase the inference time by a big factor.  <br>
 
`
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 100 -c 3d_fullres -f all -step_size 0.1
`





