import os
import argparse
import nibabel as nib
import numpy as np 
import pandas as pd
import itertools
from fsl.wrappers import fslmaths
from .misc import check_file_exists, check_fsl_installation, convert_3d_atlas_to_4d_binary_labels, filter_image

def convert_mask_to_label(mask_nifti, block_size=64, start_label=20001):
    """
    Convert a Nifti mask image to a labeled Nifti image.

    Args:
        mask_nifti (str): Path to the nifti mask image.
        block_size (int, optional): Block size. Defaults to 64.
        start_label (int, optional): Starting label. Defaults to 20001.

    Returns:
        labeled_nifti (str): Path to the labeled nifti image.
    """
    # Check FSL installation
    version = check_fsl_installation()
    print(f"FSL Version: {version}")

    # check if input files exist
    if not check_file_exists(mask_nifti):
        raise FileNotFoundError(f"Input mask file {mask_nifti} not found.")
    
    # get output file names and paths
    output_dir = os.path.dirname(mask_nifti)
    labeled_file = os.path.join(output_dir, "labeled_headmask_petfov.nii.gz")

    # Load PET image
    try:
        mask_img = nib.load(mask_nifti)
        # checking if the image shape is 3d
        if len(mask_img.shape) != 3:
            raise ValueError(f"Invalid image shape: {mask_img.shape}.\nOnly 3D mask images are supported.")
        mask_data = mask_img.get_fdata
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Error loading NIFTI file: {mask_nifti}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
    # Create an empty label image of the same shape as mask
    label_image = np.zeros_like(mask_img, dtype=np.int32)

    # Define the label counter
    label_counter = start_label

    # Iterate over the 3D image in blocks of the given size
    for x, y, z in itertools.product(range(0, mask_img.shape[0], block_size),
                                      range(0, mask_img.shape[1], block_size),
                                      range(0, mask_img.shape[2], block_size)):
        # Extract the current block
        block = mask_data[x:x+block_size, y:y+block_size, z:z+block_size]
        
        # Check if the block contains any active voxels
        if np.any(block):
            # Assign the current label to all active voxels in the block
            label_image[x:x+block_size, y:y+block_size, z:z+block_size][block > 0] = label_counter
            # Increment the label counter
            label_counter += 1

    # Create a new NIfTI image from the labeled data, using the affine from the input mask
    label_nifti = nib.Nifti1Image(label_image, affine=mask_nifti.affine)

    # Save labeled image
    try:
        nib.save(label_nifti, labeled_file)
    except IOError:
        raise IOError(f"Could not save PET sum file {labeled_file}")
    
    return labeled_file

def generate_rsfmat(image_path, label_path, output_csv_path):
    """
    Extract mean values from each label for each 3D frame in a 4D image and save as an MxN matrix.

    Args:
        image_path (str): Path to the 4D image NIfTI file.
        label_path (str): Path to the 3D label NIfTI file.
        output_csv_path (str): Path to save the output CSV file containing the mean values matrix.

    Returns:
        None
    """
    try:
        # Load the 4D image and the 3D label file
        image_4d = nib.load(image_path)
        label_3d = nib.load(label_path)
    except FileNotFoundError as fe:
        raise fe
    except nib.filebasedimages.ImageFileError as ie:
        raise ie
    except Exception as e:
        raise e

    try:
        # Get the data from the images
        image_data = image_4d.get_fdata()
        label_data = label_3d.get_fdata()
    except Exception as e:
        raise e

    # Get the unique labels excluding background (assuming label 0 is background)
    labels = np.unique(label_data)
    labels = labels[labels != 0]

    # Number of frames in the 4D image
    if len(image_data.shape) != 4:
        raise RuntimeError(f"Input Image {image_path} is not 4d.")
    num_frames = image_data.shape[3]

    # Initialize a matrix to store mean values (num_frames x num_labels)
    mean_matrix = np.zeros((num_frames, len(labels)))

    # Iterate through each frame and calculate mean values for each label
    for frame in range(num_frames):
        frame_data = image_data[:, :, :, frame]
        for i, label in enumerate(labels):
            # Extract the region corresponding to the current label
            label_mask = label_data == label
            # Calculate the mean value for the current label in the current frame
            mean_value = frame_data[label_mask].mean()
            mean_matrix[frame, i] = mean_value

    # Save the output as a matrix file (CSV format)
    try:
        df = pd.DataFrame(mean_matrix, columns=[f'Label_{int(label)}' for label in labels])
        df.to_csv(output_csv_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving CSV file: {e}")

def prepare_for_rsf(headmask_nifti, wmparc_nifti=None, sumall2t1_nifti=None):
    """
    Preprocessing for RSF computation

    Args:
        headmask_nifti (str): Path to head mask image
        wmparc_nifti (str, optional): Path to wmparc image. Defaults to None.
        sumall2t1_nifti (str, optional): Path to sumall_to_t1 image. Defaults to None.

    Raises:
        FileNotFoundError: Head Mask file not found.
        FileNotFoundError: WMPARC file not found.
        FileNotFoundError: Sumall to t1 file not found.
        RuntimeError: Error in wmparc to bin conversion.
        RuntimeError: Error in PET FOV conversion.
        RuntimeError: Error in creating headmask in PET FOV.
        RuntimeError: Error creating RSF Mask.

    Returns:
        None
    """
    output_dir = os.path.dirname(headmask_nifti)
    if wmparc_nifti is None:
        wmparc_nifti = os.path.join(output_dir, "wmparc.nii.gz")

    if sumall2t1_nifti is None:
        sumall2t1_nifti = os.path.join(output_dir, "sumall_to_t1.nii.gz")

    # check if input files exist
    if not check_file_exists(headmask_nifti):
        raise FileNotFoundError(f"Input mask file {headmask_nifti} not found.")
    
    if not check_file_exists(wmparc_nifti):
        raise FileNotFoundError(f"Input mask file {wmparc_nifti} not found.")
    
    if not check_file_exists(sumall2t1_nifti):
        raise FileNotFoundError(f"Input mask file {sumall2t1_nifti} not found.")
    
    # Create output files.
    wmparc_bin = os.path.join(output_dir, "wmparc_bin.nii.gz")
    pet_fov = os.path.join(output_dir, "petfov.nii.gz")
    head_petfov = os.path.join(output_dir, "headmask_petfov.nii.gz")
    rsfmask = os.path.join(output_dir, "RSFMask.nii.gz")
    rsfmask4d = os.path.join(output_dir,"RSFMask_4D.nii.gz")
    rsfmask4d_smth8 = os.path.join(output_dir,"RSFMask_4D_smth8mm.nii.gz")
    rsfmat_csv = os.path.join(output_dir,"rsfmat.csv")

    # Convert wmparc to bin
    try:
        _ = fslmaths(wmparc_nifti).bin().run(wmparc_bin, odt='short')
    except Exception as e:
        raise RuntimeError(f"Encounted an unexpected error wmparc to bin conversion: {e}")
    
    # Create pet fov
    try:
        _ = fslmaths(sumall2t1_nifti).bin().run(pet_fov, odt='short') 
    except Exception as e:
        raise RuntimeError(f"Encounted an unexpected error PET FOV conversion: {e}")
    
    # Create headmask in pet fov space
    try:
        _ = fslmaths(headmask_nifti).sub(wmparc_bin).mul(pet_fov).run(head_petfov, odt='int')
    except Exception as e:
        raise RuntimeError(f"Encounted an unexpected error creating headmask in PET FOV: {e}")
    
    # Convert head mask in pet fov to labeled image. 
    labeled_nifti = convert_mask_to_label(head_petfov)

    # Create RSF Mask 
    try:
        _ = fslmaths(labeled_nifti).add(wmparc_nifti).run(rsfmask, odt='int')
    except Exception as e:
        raise RuntimeError(f"Encounted an unexpected error creating RSF Mask: {e}")
    
    # Convert 3d RSF label Mask to 4d binary masks
    convert_3d_atlas_to_4d_binary_labels(rsfmask, rsfmask4d)

    # Smooth to 4d RSF binary mask with 8x8x8 gaussian filter.
    filter_image(rsfmask4d, rsfmask4d_smth8)

    # Create RSF matrix
    generate_rsfmat(rsfmask4d_smth8, rsfmask, rsfmat_csv)

    return None

def main():
    parser = argparse.ArgumentParser(description='Prepare for RSF correction.')
    parser.add_argument('--headmask', type=str, help='Input Head Mask (nii.gz format).', required=True)
    parser.add_argument('--wmparc', type=str, default=None, help='Input WMPARC file (nii.gz format, optional).', required=False)
    parser.add_argument('--sumall_t1', type=str, default=None, help='Path to the sumall to T1 (nii.gz format, optional).', required=False)
    args = parser.parse_args()

    prepare_for_rsf(args.headmask, args.wmparc, args.sumall_t1)
    print("Prepare for RSF correction Done.")

if __name__ == "__main__":
    main()
