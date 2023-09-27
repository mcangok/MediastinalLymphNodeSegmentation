# MediastinalLymphNodeSegmentation Algorithm

Challenge Submission of the CompAI team

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

Preprocess data with nnUNetv2 (plans are already in nnUNet_preprocessed):  <br>

`
nnUNetv2_preprocess -d 100 -c 3d_fullres -np 16
`

<br>

`
nnUNetv2_train 100 3d_fullres all
`


## Inference with model

After training you can use the model as a standard nnUNetv2 model:  <br>

`
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 100 -c 3d_fullres -f all
`

In my docker submission i reduced the step size of the patch-based inference, such that there is a bigger ensemble effect. I think that leads to a more stable output, but will increase the inference time by a big factor.  <br>
 
`
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 100 -c 3d_fullres -f all -step_size 0.1
`





