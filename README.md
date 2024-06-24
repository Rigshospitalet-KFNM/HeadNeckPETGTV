# PET GTV pipeline

This is a pipeline made for $Citation needed

## Installation

Ensure you have podman and Nvidia container toolkit installed.
Contact repo owner for docker image

Start with: 
`git clone https://github.com/Rigshospitalet-KFNM/HeadNeckPETGTV --recurse-submodules`

Install virtual environment

`python3 -m venv venv`
`source venv/bin/activate`
`pip install submodules/dicom_node`

Then install dcm2niix and niftyreg found in the submodules

### Environment variables 

This pipeline is controlled by environment variables, set them using

`export NAME=VALUE`

* PIPELINE_ARCHIVE_PATH - Path, where the dicom node stores images temporary
* PIPELINE_WORKING_PATH - Path, where the dicom node executes the process function
* PIPELINE_LOG_PATH - Path, to the log file
* PIPELINE_DCM2NIIX - Path, to the dcm2niix executable 
* PIPELINE_RESAMPLE - Path,  to the reg_resample executable
* PIPELINE_AE_TITLE - Optional List of str, comma seperated, spaces are stripped. Accepted AE title, if present only ae titles in list is accepted
* PIPELINE_SEGMENTATION_ARCHIVE - Optional Path of directory, if present segmentation are saved to this directory.
