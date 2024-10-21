import os
import fsl.wrappers as fsl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import nibabel as nib
import sys
import pandas as pd
from .misc import check_fsl_installation

def perform_motion_correction(input_4d_nifti, output_dir=None):
    """
    Perform PET motion correction using FSL's mcflirt and generate a motion parameter plot.

    Parameters:
    - input_4d_nifti (str): Path to the input 4D NIfTI image.
    - output_dir (str): Path to the output directory (optional). Default is None

    Returns:
    - output_4d_nifti (str): Path to the motion-corrected 4D NIfTI image.
    """
    # Checking if FSL is installed and configured
    version = check_fsl_installation()
    print(f"FSL Version: {version}")

    # Check if input image exists
    if not os.path.isfile(input_4d_nifti):
        raise FileNotFoundError(
            f"Input image file '{input_4d_nifti}' not found")

    # Prepare output file paths
    if output_dir is None:
        output_dir = os.path.dirname(input_4d_nifti)
    os.makedirs(output_dir, exist_ok=True)
    input_filename = os.path.basename(input_4d_nifti)

    output_4d_nifti = os.path.join(
        output_dir, input_filename.split('.')[0] + '_moco')
    motion_params_file = os.path.join(
        output_dir, input_filename.split('.')[0] + '_moco.par')
    motion_params_plot = os.path.join(
        output_dir, input_filename.split('.')[0] + '_moco.png')
    
    # Load the 4D NIfTI image
    try:
        img = nib.load(input_4d_nifti)
        # checking if the image shape is 3d or 4d
        if len(img.shape) not in [3, 4]:
            raise ValueError(f"Invalid image shape: {img.shape}.\nOnly 3D or 4D images are supported.")
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Error loading NIFTI file: {input_4d_nifti}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
    if len(img.shape) == 3:
        # If the number of volumes is 1, create a copy of the input file as the output file
        try:
            _ = fsl.fslmaths(input_4d_nifti).mul(1).run(output_4d_nifti)
        except Exception as e:
            print(f"ERROR: {e}")
        print(
            f"Single volume detected. Created a copy of the input file as: {output_4d_nifti}")
        return output_4d_nifti
        sys.exit(0)

    # Run mcflirt
    try:
        _ = fsl.mcflirt(infile=input_4d_nifti, out=output_4d_nifti,
                        plots='plots', meanvol='meanvol', report='report')
    except Exception as e:
        print(f"MCFLIRT ERROR: {e}")

    # Check if motion corrected nifti file was generated
    if not os.path.isfile(output_4d_nifti + '.nii.gz'):
        raise FileNotFoundError(
            f"Motion corrected file not found: {output_4d_nifti}.nii.gz")

    # Check if motion parameter file was generated
    if not os.path.exists(motion_params_file):
        raise FileNotFoundError(
            f"Motion parameter file not found: {motion_params_file}")

    # Load motion parameters
    motion_params = np.loadtxt(motion_params_file)

    # Create figure with subplots
    plt.figure(figsize=(10, 12))

    # Subplot for translational motion parameters
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(motion_params[:, 3:6])
    plt.xlabel('Volume')
    plt.ylabel('Translation (mm)')
    plt.title('Translational Motion Parameters from MCFLIRT')
    plt.legend(['X (mm)', 'Y (mm)', 'Z (mm)'])

    # Subplot for rotational motion parameters
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.plot(motion_params[:, 0:3])
    plt.xlabel('Volume')
    plt.ylabel('Rotation (radians)')
    plt.title('Rotational Motion Parameters from MCFLIRT')
    plt.legend(['Rot X (radians)', 'Rot Y (radians)', 'Rot Z (radians)'])

    # Save the plot
    plt.savefig(motion_params_plot)
    plt.close()

    print(
        f"Motion correction completed. Corrected image saved to: {output_4d_nifti}")
    print(f"Motion parameters plot saved to: {motion_params_plot}")

    return output_4d_nifti


def main():
    parser = argparse.ArgumentParser(
        description="Perform PET motion correction using FSL's mcflirt.")
    parser.add_argument('-i','--input_4d_nifti', type=str,
                        help='Path to the input 4D NIfTI image.', required=True)
    parser.add_argument('-od', '--output_dir', type=str, help='Path to the output directory', default=None, required=False)

    args = parser.parse_args()

    # Call function to perform motions correction
    perform_motion_correction(args.input_4d_nifti, args.output_dir)


if __name__ == "__main__":
    main()
