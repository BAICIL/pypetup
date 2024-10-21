import os
import argparse
from .misc import run_command, check_file_exists, check_freesurfer_installation

def convert_mgz_to_nii(T1_mgz, wmparc_mgz, output_dir):
    """
    Converts T1.mgz and wmparc.mgz files to .nii format using FreeSurfer's `mri_convert` command.
    
    Args:
        T1_mgz (str): Path to the T1.mgz file.
        wmparc_mgz (str): Path to the wmparc.mgz file.
        output_dir (str): Path to the output directory where .nii files will be saved.
    
    Raises:
        FileNotFoundError: If T1.mgz or wmparc.mgz is not found.
        OSError: If there is an issue creating the output directory.
    """
    # check if freesurfer is installed and get version information
    version = check_freesurfer_installation()
    print(f"FreeSurfer Version: {version}")

    # Check if the input files exist
    if not check_file_exists(T1_mgz):
        raise FileNotFoundError(f"T1 file not found: {T1_mgz}")
    if not check_file_exists(wmparc_mgz):
        raise FileNotFoundError(f"wmparc file not found: {wmparc_mgz}")

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed not create output directory {output_dir}.\n{e}")

    # Generate the output file paths
    T1_nii = os.path.join(output_dir, 'T1.nii.gz')
    wmparc_nii = os.path.join(output_dir, 'wmparc.nii.gz')
    head_mask = os.path.join(output_dir, 'HeadMask.nii.gz')

    # Construct FreeSurfer mri_convert commands
    command_T1 = ['mri_convert', T1_mgz, T1_nii, '--in_orientation', 'LIA', '--out_orientation', 'LAS']
    command_wmparc = ['mri_convert', wmparc_mgz, wmparc_nii, '--in_orientation', 'LIA', '--out_orientation', 'LAS']
    command_headmask = ['mri_seghead', '--invol', T1_nii, '--outvol', head_mask, '--thresh', '20', '--fill-holes-islands']

    # Execute the conversion commands
    print(f"Converting {T1_mgz} to {T1_nii}...")
    run_command(command_T1)

    print(f"Converting {wmparc_mgz} to {wmparc_nii}...")
    run_command(command_wmparc)

    print(f"Generating head mask...")
    run_command(command_headmask)

    return None


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert T1.mgz and wmparc.mgz to NIfTI format and generate head mask using FreeSurfer.")
    parser.add_argument('--t1', type=str, help="Path to the T1.mgz file", required=True)
    parser.add_argument('--wmparc', type=str, help="Path to the wmparc.mgz file", required=True)
    parser.add_argument('--output_dir', type=str, help="Path to the output directory where NIfTI files will be saved", required=True)

    # Parse arguments
    args = parser.parse_args()

    # Convert .mgz to .nii
    try:
        convert_mgz_to_nii(args.t1, args.wmparc, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
