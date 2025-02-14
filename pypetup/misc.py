# import required modules
import json
import os
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def check_file_exists(filepath):
    """
    Check if a file exists at the specified path.

    Args:
    filepath (str): The path to the file.

    Returns:
    bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(filepath)


def save_nifti(image, affine, output_path):
    """
    Save a NIfTI image to a file.

    Parameters:
    image (numpy.ndarray): The image data array. This is typically a 3D or 4D numpy array.
    affine (numpy.ndarray): The affine transformation matrix. This is a 4x4 matrix that defines the spatial orientation of the image data.
    output_path (str): The file path where the NIfTI image will be saved.

    Returns:
    None

    Raises:
    ValueError: If the image or affine are not valid.
    IOError: If the file could not be written.
    """
    nib.openers.Opener.default_compresslevel = 9
    try:
        nib.save(nib.Nifti1Image(image, affine), output_path)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid input for image or affine: {e}")
    except IOError as e:
        raise IOError(f"Could not write file to {output_path}: {e}")

    return None


def convert_3d_atlas_to_4d_binary_labels(atlas_path, output_4d_path):
    """
    Converts a 3D atlas NIfTI file into a 4D binary labels NIfTI file.

    This function takes a 3D atlas file, where each voxel's value represents a
    region of interest (ROI), and converts it into a 4D binary labels file.
    Each 3D volume in the 4D file corresponds to a binary mask of a specific ROI.

    Args:
        atlas_path (str): The file path to the input 3D atlas NIfTI file.
        output_4d_path (str): The file path to save the output 4D binary labels NIfTI file.

    Returns:
        None

    Example:
        Given an atlas file with labels for different brain regions, this function
        will create a 4D NIfTI file where each 3D volume is a binary mask for one
        of the regions.

    Notes:
        - The function uses compression level 9 for saving the output NIfTI file.
        - The background label is assumed to be 0 and is excluded from the binary masks.
    """

    nib.openers.Opener.default_compresslevel = 9

    # Load the 3D atlas NIfTI file
    try:
        atlas_img = nib.load(atlas_path)
        if len(atlas_img.shape) != 3:
            raise ValueError(
                f"Invalid image shape: {atlas_img.shape}.\nOnly 3D atlas images are supported."
            )
        atlas_data = atlas_img.get_fdata()
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Error loading NIFTI file: {atlas_path}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    # Get unique ROI labels from the atlas (excluding the background label, assumed to be 0)
    roi_labels = np.unique(atlas_data)
    #roi_labels = roi_labels[roi_labels != 0]

    # Create an empty 4D array with the same spatial dimensions as the atlas and depth equal to the number of ROIs
    shape_4d = list(atlas_data.shape) + [len(roi_labels)]
    binary_4d_data = np.zeros(shape_4d, dtype=np.uint8)

    # Vectorized operation to create binary label maps for each ROI
    for i, label in enumerate(roi_labels):
        binary_4d_data[..., i] = atlas_data == label
    # Create a new NIfTI image for the 4D data
    binary_4d_img = nib.Nifti1Image(
        binary_4d_data, affine=atlas_img.affine, header=atlas_img.header
    )

    # Save labeled image
    try:
        nib.save(binary_4d_img, output_4d_path)
    except IOError:
        raise IOError(f"Could not save PET sum file {output_4d_path}")

    print(f"Conversion complete. Saved as {output_4d_path}")

    return None


def check_img_orientation(affine, desired_orientation=("L", "A", "S")):
    """
    Check if the given image's orientation matches the desired orientation.

    Parameters:
    affine (numpy.ndarray): The affine matrix of the input image.
    desired_orientation (tuple of str, optional): The desired orientation in terms of axis codes. Default is ('L', 'A', 'S').

    Returns:
    bool: True if the image's orientation matches the desired orientation, False otherwise.

    Raises:
    ValueError: If the affine is not a valid numpy array.
    """
    if not isinstance(affine, np.ndarray):
        raise ValueError("Affine must be a numpy ndarray.")

    try:
        img_orientation = nib.aff2axcodes(affine)
        return img_orientation == desired_orientation
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while checking the image orientation: {e}"
        )


def reorient_image(img, desired_orientation=("L", "A", "S")):
    """
    Reorient a given image to the desired orientation.

    Parameters:
    img (nibabel image object): The input image to be reoriented.
    desired_orientation (tuple of str, optional): The desired orientation in terms of axis codes. Default is ('L', 'A', 'S').

    Returns:
    tuple: A tuple containing the reoriented image data as a numpy array (dtype float32) and the new affine matrix.

    Raises:
    ValueError: If the input image is not a valid nibabel image object.
    """
    if not isinstance(img, nib.Nifti1Image):
        raise ValueError("Input image must be a nibabel Nifti1Image object.")

    try:
        img_orientation = nib.orientations.io_orientation(img.affine)
        out_orientation = nib.orientations.axcodes2ornt(desired_orientation)
        transform = nib.orientations.ornt_transform(img_orientation, out_orientation)
        reoriented_data = nib.orientations.apply_orientation(
            img.get_fdata(), transform
        ).astype(np.float32)
        reoriented_affine = img.affine @ nib.orientations.inv_ornt_aff(
            transform, img.shape
        )
        return reoriented_data, reoriented_affine
    except Exception as e:
        raise RuntimeError(f"An error occurred while reorienting the image: {e}")


def check_fsl_installation():
    """
    Check if FSL is installed.
    Raises an error if FSL is not found.
    Return:
        version (str): Version of FSL installed.
    """
    from fsl.utils.platform import Platform

    try:
        version = Platform().fslVersion
        return version
    except Exception as e:
        raise RuntimeError(
            f"FSL software not found. Please ensure FSL is installed and properly configured.\n{e}"
        )


def check_freesurfer_installation():
    """
    Check if FreeSurfer is installed and if it has a valid license file.
    Raise an error if FreeSurfer is not install or configured.
    Return:
        version (str): Version of FreeSurfer installed.
    """
    try:
        # Check if FreeSurfer command is available and grab the version number
        version = subprocess.run(
            ["recon-all", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if not version:
            raise Exception("Failed to retrieve FreeSurfer version.")
    except subprocess.CalledProcessError:
        raise Exception("FreeSurfer is not installed or not in the PATH.")

    # Check if the FREESURFER_HOME environment variable is set
    freesurfer_home = os.getenv("FREESURFER_HOME")
    if not freesurfer_home:
        raise Exception("FREESURFER_HOME environment variable is not set.")

    # Check if the license file exists in any of the possible locations
    license_paths = [
        os.path.join(freesurfer_home, "license.txt"),
        os.path.join(freesurfer_home, ".license"),
        os.getenv("FS_LICENSE"),
    ]

    if not any(
        license_path and os.path.isfile(license_path) for license_path in license_paths
    ):
        raise Exception(
            "FreeSurfer license file not found in any of the expected locations."
        )

    return version


def get_output_filename(input_filename, old_suffix=None, new_suffix=""):
    """
    Generate an output image filename based on the input image filename and a suffix.
    It assumes that the extension is nii.gz.

    Args:
    input_filename (str): Input filename.
    old_suffix (str, optional): Old suffix to replace, if present.
    suffix (str): Suffix to replace '_moco' or append before the extension.

    Returns:
    str: Output filename
    """

    if old_suffix and old_suffix in input_filename:
        return input_filename.replace(old_suffix, new_suffix)
    else:
        return input_filename.replace(".nii.gz", new_suffix + ".nii.gz")


def run_command(command):
    """
    Executes a given command using subprocess module and handles exceptions.

    Args:
        command (str or list): The command to be executed. It can be a string or a list of strings.

    Returns:
        returncode (int): The return code of the executed command.

    Raises:
        ValueError: If the command is not provided as a string or list.
        RuntimeError: If the command execution fails (non-zero return code).
    """
    if not isinstance(command, list):
        raise ValueError("Command must be a list of strings.")

    try:
        # Run the command and capture stdout, stderr, and return code
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed to execute with return code {result.returncode}"
            )

        # Return command output details
        return None

    except subprocess.CalledProcessError as e:
        # Handle case where the command fails with a non-zero exit status
        raise RuntimeError(f"Command failed with error: {e.stderr}") from e
    except FileNotFoundError:
        # Handle the case where the command is not found
        raise RuntimeError(f"Command not found: {command}")
    except Exception as e:
        # Handle any other exception
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")


def load_json(json_file):
    """
    Load a JSON file and return its contents.

    Parameters:
    json_file (str): Path to the JSON file.

    Returns:
    dict: Parsed JSON data.

    Raises:
    FileNotFoundError: If the provided JSON file path does not exist.
    json.JSONDecodeError: If the JSON file is malformed.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"The file {json_file} does not exist.")

    with open(json_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing the JSON file: {e}")

    return data


def filter_image(image_4d, output_4d, filter_size=[8, 8, 8]):
    """
    This function filters the image so that the effect smoothing is 8x8x8 mm3. This is done to
    harmonize the data across different scanners.

    Raises:
        fe: FileNotFound Error
        ie: ImageFileError from nibabel exceptions.
        e: Other Exceptions
        ValueError: Error in input values
    """
    if not check_file_exists(image_4d):
        raise FileNotFoundError(f"The input file {image_4d} not found.")

    try:
        img = nib.load(image_4d)
        data = img.get_fdata(dtype=np.float32)
    except nib.filebasedimages.ImageFileError as ie:
        raise ie
    except Exception as e:
        raise e
    fwhm = np.array(filter_size)
    sigma = fwhm / np.sqrt(8 * np.log(2))
    voxel_size = np.abs(np.diag(img.affine))[:3]
    sigma_voxel = sigma / voxel_size
    try:
        if data.ndim == 3:
            smoothed_data = gaussian_filter(data, sigma=sigma_voxel)
        elif data.ndim == 4:
            smoothed_data = np.zeros_like(data)
            for t in range(data.shape[3]):
                smoothed_data[:, :, :, t] = gaussian_filter(
                    data[:, :, :, t], sigma=sigma_voxel
                )
        else:
            raise ValueError("Input NIFTI image must be 3D or 4D.")
    except Exception as e:
        raise e
    try:
        smoothed_img = nib.Nifti1Image(smoothed_data, img.affine, img.header)
        nib.save(smoothed_img, output_4d)
    except nib.filebasedimages.ImageFileError as ie:
        raise ie
    except Exception as e:
        raise e

    return None


def write_dataframe_to_csv(df, output_file):
    """_summary_

    Args:
        df (pandas.DataFrame): Pandas data frame to write to file.
        output_file (str): Path with file name to write the data frame to.

    Raises:
        IOError: Error writing file to disk

    Returns:
        None
    """
    try:
        df.to_csv(output_file, sep="\t", index=False)
    except Exception as e:
        raise IOError(f"Error saving DataFrame to CSV: {e}")
    return None


def copy_file(src, dest):
    try:
        # Check if the source file exists
        if not os.path.exists(src):
            print(f"Source file does not exist: {src}")
            return

        # Check if the destination file already exists
        if not os.path.exists(dest):
            shutil.copy2(src, dest)
            print(f"Copied {src} to {dest}")
        else:
            print(f"File already exists at {dest}. Skipping copy.")
        return None
    except PermissionError:
        print(f"Permission denied when accessing {src} or {dest}.")

    except FileNotFoundError:
        print(f"File not found error: either {src} or {dest} path is invalid.")

    except shutil.SameFileError:
        print("Source and destination represent the same file.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def time_function(func, *args, **kwargs):
    """
    Times the execution of a given function with specified arguments and prints
    the execution time in hh:mm:ss format.

    Parameters:
    - func (callable): The function to time.
    - *args: Positional arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.

    Returns:
    - result: The result of the function call if successful, otherwise None.

    Prints:
    - The execution time in hh:mm:ss format.
    - An error message if the function execution fails.
    """
    try:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculate elapsed time
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Print the execution time
        print(
            f"Execution time for {func.__name__}: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        return result

    except Exception as e:
        raise RuntimeError(f"Error occurred while executing {func.__name__}: {e}")
