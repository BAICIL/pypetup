import argparse
import os

import nibabel as nib
import numpy as np
from fsl.wrappers import fslmaths

from .misc import (
    check_file_exists,
    check_fsl_installation,
    convert_3d_atlas_to_4d_binary_labels,
    filter_image,
)


def convert_mask_to_label(mask_nifti_file, block_size=64, start_label=20001):
    """
    Convert a NIfTI mask image to a labeled NIfTI image.

    Parameters:
    mask_nifti_file (str): Path to the NIfTI image representing the mask.
    block_size (int): Size of the block to iterate over (default is 64).
    start_label (int): Starting integer label for the active voxels (default is 20001).

    Returns:
    nibabel.Nifti1Image: A labeled NIfTI image.
    """
    # check if input files exist
    if not check_file_exists(mask_nifti_file):
        raise FileNotFoundError(f"Input mask file {mask_nifti_file} not found.")

    # get output file names and paths
    output_dir = os.path.dirname(mask_nifti_file)
    labeled_file = os.path.join(output_dir, "labeled_headmask_petfov.nii.gz")

    # Load the mask nifti file
    try:
        mask_nifti = nib.load(mask_nifti_file)
        if len(mask_nifti.shape) != 3:
            raise ValueError(
                f"Invalid image shape: {mask_nifti.shape}.\nOnly 3D mask images are supported."
            )
        mask = mask_nifti.get_fdata()
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Error loading NIFTI file: {mask_nifti_file}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    # Create an empty label image of the same shape as mask
    label_image = np.zeros_like(mask, dtype=np.int32)

    # Define the label counter
    label_counter = start_label

    # Get the shape of the mask
    x_max, y_max, z_max = mask.shape

    # Iterate over the 3D image in blocks of the given size
    for x in range(0, x_max, block_size):
        for y in range(0, y_max, block_size):
            for z in range(0, z_max, block_size):
                # Define the ending indices for the current block
                x_end = min(x + block_size, x_max)
                y_end = min(y + block_size, y_max)
                z_end = min(z + block_size, z_max)

                # Extract the current block
                block = mask[x:x_end, y:y_end, z:z_end]

                # Check if the block contains any active voxels
                if np.any(block):
                    # Assign the current label to all active voxels in the block
                    label_image[x:x_end, y:y_end, z:z_end][block > 0] = label_counter
                    # Increment the label counter
                    label_counter += 1

    # Create a new NIfTI image from the labeled data, using the affine from the input mask
    labeled_nifti = nib.Nifti1Image(label_image, affine=mask_nifti.affine)

    # Save labeled image
    try:
        nib.save(labeled_nifti, labeled_file)
    except IOError:
        raise IOError(f"Could not save PET sum file {labeled_file}")

    return labeled_file


def generate_rsfmat(image_path, label_path, output_dir=None, batch_size=50):
    """
    Extract mean values from each label for each 3D frame in a 4D image and save as an MxN matrix.

    Args:
        image_path (str): Path to the 4D image NIfTI file.
        label_path (str): Path to the 3D label NIfTI file.
        output_csv_path (str): Path to save the output CSV file containing the mean values matrix.
        batch_size(int): Number of frames to process at a time (default = 50)

    Returns:
        None
    """

    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    output_file = os.path.join(output_dir, "rsfmat.txt")

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
        image_data_proxy = image_4d.dataobj
        label_data = label_3d.get_fdata()
    except Exception as e:
        raise e

    # Get the unique labels in the atlas
    unique_labels = np.unique(label_data)
    #unique_labels = unique_labels[unique_labels != 0]  # Exclude the background label

    # Initialize a list to store the mean values for each label across all frames
    mean_values = {label: [] for label in unique_labels}

    # Iterate over the frames in batches
    n_frames = image_4d.shape[-1]
    for start_idx in range(0, n_frames, batch_size):
        # Determine the end index for the current batch
        end_idx = min(start_idx + batch_size, n_frames)

        # Load the current batch of frames
        time_slices = image_data_proxy[..., start_idx:end_idx]

        #Process each label for the current batch
        for label in unique_labels:
            print(f"Processing label: {label} for frames: {start_idx} - {end_idx}")
            roi_mask = label_data == label
            roi_values = time_slices[roi_mask, :] # Extract values across all batch frames for current roi
            mean_roi_values = np.mean(roi_values, axis=0) # Mean across the voxels for each frame in the batch
            mean_values[label].extend(mean_roi_values) # Accumulate mean values for this label

    # Prepare the mean values matrix
    mean_values_matrix = []
    for label in unique_labels:
        mean_values_matrix.append(mean_values[label])

    # Convert mean values to numpy array
    mean_values_matrix = np.array(mean_values_matrix)

    # Save the rsfmat to a text file
    np.savetxt(output_file, mean_values_matrix, delimiter="\t", fmt='%6e')
    return None


def prepare_for_rsf(headmask_nifti, wmparc_nifti=None, batch_size=50):
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
    # Check FSL installation
    version = check_fsl_installation()
    print(f"FSL Version: {version}")

    output_dir = os.path.dirname(headmask_nifti)
    if wmparc_nifti is None:
        wmparc_nifti = os.path.join(output_dir, "wmparc.nii.gz")

    # check if input files exist
    if not check_file_exists(headmask_nifti):
        raise FileNotFoundError(f"Input mask file {headmask_nifti} not found.")

    if not check_file_exists(wmparc_nifti):
        raise FileNotFoundError(f"Input mask file {wmparc_nifti} not found.")

    # Create output files.
    wmparc_bin = os.path.join(output_dir, "wmparc_bin.nii.gz")
    head_nifti = os.path.join(output_dir, "head.nii.gz")
    rsfmask = os.path.join(output_dir, "RSFMask.nii.gz")
    rsfmask4d = os.path.join(output_dir, "RSFMask_4D.nii.gz")
    rsfmask4d_smth8 = os.path.join(output_dir, "RSFMask_4D_g8.nii.gz")

    # Convert wmparc to bin
    try:
        _ = fslmaths(wmparc_nifti).bin().run(wmparc_bin, odt="short")
    except Exception as e:
        raise RuntimeError(
            f"Encounted an unexpected error wmparc to bin conversion: {e}"
        )

    # Create head mask without brain
    try:
        _ = (
            fslmaths(headmask_nifti)
            .sub(wmparc_bin)
            .run(head_nifti, odt="int")
        )
    except Exception as e:
        raise RuntimeError(
            f"Encounted an unexpected error creating headmask: {e}"
        )

    # Convert head mask in pet fov to labeled image.
    labeled_file = convert_mask_to_label(head_nifti)

    # Create RSF Mask
    try:
        _ = fslmaths(labeled_file).add(wmparc_nifti).run(rsfmask, odt="int")
    except Exception as e:
        raise RuntimeError(f"Encounted an unexpected error creating RSF Mask: {e}")

    # Convert 3d RSF label Mask to 4d binary masks
    convert_3d_atlas_to_4d_binary_labels(rsfmask, rsfmask4d)

    # Smooth to 4d RSF binary mask with 8x8x8 gaussian filter.
    filter_image(rsfmask4d, rsfmask4d_smth8)

    # Create RSF matrix
    generate_rsfmat(rsfmask4d_smth8, rsfmask, batch_size=batch_size)

    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare for RSF correction.")
    parser.add_argument(
        "--headmask", type=str, help="Input Head Mask (nii.gz format).", required=True
    )
    parser.add_argument(
        "--wmparc",
        type=str,
        default=None,
        help="Input WMPARC file (nii.gz format, optional).",
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for processing RSFMask 4D frames in batches (default = 50).",
        required=False,
    )
    args = parser.parse_args()

    prepare_for_rsf(args.headmask, args.wmparc, args.batch_size)
    print("Prepare for RSF correction Done.")


if __name__ == "__main__":
    main()
