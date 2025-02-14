### pypetup
PET Unified Pipeline in Python.

This package perform quantative processing of PET data. This package closely replicates the original PET Unified Pipeline (PUP) developed by Dr. Yi Su (https://github.com/ysu001/PUP).

# Pre-requisites 
* OS = MacOS or Linux
* FreeSurfer >= version 7 correctly configured
* FSL >= version 6 correctly configured

# Install

Clone the github repo:

```
pip install git+https://github.com/BAICIL/pypetup.git
```

# Usage
```
run_pup [-h] --pet_nifti PET_NIFTI 
            [--pet_json PET_JSON] 
            [--derivatives_dir DERIVATIVES_DIR] 
            --fs_dir FS_DIR 
            [--t1_filename T1_FILENAME]
            [--batch_size BATCH_SIZE]
            [--start_time START_TIME] 
            [--duration DURATION] 
            [--norsf]

Process PET data using the PUP workflow.

options:
  -h, --help            
    show this help message and exit
  --pet_nifti PET_NIFTI 
    Path to the input PET file.
  --pet_json PET_JSON
    Path to the input PET JSON file.
  --derivatives_dir DERIVATIVES_DIR
    Path to the base directory (optional, default=None).
  --fs_dir FS_DIR
    Path of the Subjects FreeSurfer mri directory.
  --t1_filename T1_FILENAME
    File name with extension of the T1 file (optional, default=orig_nu.mgz)
  --batch_size BATCH_SIZE
    Batch size for processing 4D RSFMask image (optional, default=50)
  --start_time START_TIME
    Start time for frames of interest (optional, default=None)
  --duration DURATION
    Total duration for frames of interest (optional, default=None)
  --norsf
    Do not perform RSF correction
```

# Example
1. Minimum inputs: In this case the derivatives directory will be created in the directory where the PET input data resides. The json file is assumed to be in the same location as the PET file with similar name as the nifti. All frames will be used as frames of interest. RSF PVC correction will be attempted.

```
run_pup --pet_nift /path/to/pet.nii.gz \
        --fs_dir /path/to/subject/mri 

```
2. Organized minimal inputs: In this case the derivatives directory will be provided by the user to better organize the output. In this case, the processed data will be located in the `derivatives/sub-XXX/ses-XXX/Tracer/`

```
run_pup --pet_nifti /path/to/pet.nii.gz \
        --derivatives_dir /path/to/derivatives
        --fs_dir /path/to/subject/mri
