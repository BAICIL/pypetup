import os
import argparse
import nibabel as nib
import numpy as np
from fsl.wrappers import flirt, applyxfm
from .misc import check_fsl_installation, get_output_filename, check_file_exists

def sum_4d_pet(pet_image):
    """
    Sum a 4D PET image to create a 3D sum image.
    
    Args:
    pet_image (nibabel.Nifti1Image): 4D PET image
    
    Returns:
    nibabel.Nifti1Image: 3D summed PET image
    """
    pet_data = pet_image.get_fdata()
    sum_data = np.sum(pet_data, axis=3)
    return nib.Nifti1Image(sum_data, pet_image.affine, pet_image.header)

def coregister_pet_to_t1(pet_file, t1_file, output_dir=None):
    """
    Coregister PET image to T1 image using FSL FLIRT.
    
    Args:
        pet_file (str): Path to the input PET file
        t1_file (str): Path to the input T1 file
        output_file (str): Path to the output coregistered PET file (option). default=None

    Returns:
        
    """
    # Check FSL installation
    version = check_fsl_installation()
    print(f"FSL Version: {version}")

    # check if input files exist
    if not check_file_exists(pet_file):
        raise FileNotFoundError(f"Input PET file {pet_file} not found.")
    if not check_file_exists(t1_file):
        raise FileNotFoundError(f"Input T1 file {t1_file} not found.")
    
    # get input and output file names and paths
    if output_dir is None:
        output_dir = os.path.dirname(pet_file)
    os.makedirs(output_dir, exist_ok=True)
    pet_filename = os.path.basename(pet_file)
    pet_sum_filename = get_output_filename(pet_filename, '_moco', '_sumall')
    pet_coreg_filename = get_output_filename(pet_filename, '_moco', '_coregt1')
    pet_sum_file = os.path.join(output_dir, pet_sum_filename)
    pet_coreg_file = os.path.join(output_dir, pet_coreg_filename)

    # Load PET image
    try:
        pet_img = nib.load(pet_file)
        # checking if the image shape is 3d or 4d
        if len(pet_img.shape) not in [3, 4]:
            raise ValueError(f"Invalid image shape: {pet_img.shape}.\nOnly 3D or 4D images are supported.")
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Error loading NIFTI file: {pet_file}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
    # Sum 4D PET image to create 3D sum image
    pet_sum_image = sum_4d_pet(pet_img)
    try:
        nib.save(pet_sum_image, pet_sum_file)
    except IOError:
        raise IOError(f"Could not save PET sum file {pet_sum_file}")
    
    # Coregister PET sum image to T1 using FLIRT
    matrix_file = os.path.join(output_dir, "sumall_to_t1.mat")
    coregistered_file = os.path.join(output_dir, "sumall_to_t1.nii.gz")
    print("Computing PET to T1 transformation...")
    try:
        flirt(src=pet_sum_file, ref=t1_file, out=coregistered_file, omat=matrix_file)
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during FLIRT: {e}")
    
    # Apply transformation to the original 4D PET image
    print("Applying transformation to PET image...")
    try:
        applyxfm(src=pet_file, ref=t1_file, mat=matrix_file, out=pet_coreg_file, interp='spline')
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during APPLYXFM: {e}")
    
    return pet_coreg_file

def main():
    parser = argparse.ArgumentParser(description='Coregister PET image to T1 image using FSL')
    parser.add_argument('--pet_file', type=str, help='Input PET image file (nii.gz format)', required=True)
    parser.add_argument('--t1_file', type=str, help='Input T1 image file (nii.gz format)', required=True)
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory (optional).', required=False)
    args = parser.parse_args()

    # Perform coregistration
    output = coregister_pet_to_t1(args.pet_file, args.t1_file, args.output_dir)
    print(f"Coregister PET: {output}")

if __name__ == "__main__":
    main()
