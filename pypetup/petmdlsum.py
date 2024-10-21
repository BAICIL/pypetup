import argparse
import nibabel as nib
import numpy as np
import os
import shutil
from .misc import get_output_filename, check_file_exists

def decay_correction(time, start_time, half_life):
    """
    Computes the adjusted decay correction factor for the averaged sum image. 

    Args:
        time (float): mid point time in minutes.
        start_time (float): start time in minutes.
        half_life (float): tracer half life in seconds.

    Returns:
        float: computed decay factor for the averaged sum image. 
    """
    # time and start_time are in minutes, half_life is in seconds
    decay_constant = np.log(2) / half_life
    return np.exp(decay_constant * 60 * (time - start_time))  # Convert minutes to seconds

def model_sum_pet(input_file, start_frame, end_frame, half_life, start_times, durations, decay_corrections, scale_factor=1.0, use_first_decay=False, output_dir=None):
    """_summary_

    Args:
        input_file (str): Path to the input image. 
        start_frame (int): Model Start frame.
        end_frame (int): Model End frame. 
        half_life (float): Tracer half life in seconds.
        start_times (list): List of frame start times.
        durations (list): List of frame durations. 
        decay_corrections (list): List of frame decay correction factors.
        scale_factor (float, optional): Scaling factor for image. Defaults to 1.0.
        use_first_decay (bool, optional): Optional use the decay correction factor of the first frame. Defaults to False.
        output_dir (str, optional): Path of the output directory. Defaults to None.

    Returns:
        str: path of output file.
    """
    
    # check if input files exist
    if not check_file_exists(input_file):
        raise FileNotFoundError(f"Input PET file {input_file} not found.")
    
    # get input and output file names and paths
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    input_filename = os.path.basename(input_file)
    output_filename = get_output_filename(input_filename, '_coregt1', '_msum')
    output_file = os.path.join(output_dir, output_filename)
    
    # Load PET image
    try:
        img = nib.load(input_file)
        data = img.get_fdata()
        # checking if the image shape is 3d or 4d
        if len(img.shape) not in [3, 4]:
            raise ValueError(f"Invalid image shape: {img.shape}.\nOnly 3D or 4D images are supported.")
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Error loading NIFTI file: {input_file}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
    # Get the number of frames
    n_frames = data.shape[3] if len(data.shape) > 3 else 1
    
    if n_frames == 1:
        print("Single Frame Data")
        try:
            shutil.copy(input_file, output_file)
        except Exception as e:
            print(f"An error occurred while copying the file: {e}")
        return output_file
    
    if start_frame < 1 or end_frame > n_frames:
        raise ValueError(f"Invalid frame range. Total frames: {n_frames}")
    
    # Initialize variables
    summed_data = np.zeros(data.shape[:3])
    total_duration = 0
    start_time = None
    first_decay_cor = None

    for frame in range(start_frame, end_frame + 1):
        frame_data = data[..., frame]
        frame_duration = durations[frame]  # in minutes
        frame_start = start_times[frame]   # in minutes
        decay_cor = decay_corrections[frame]
        
        if start_time is None:
            start_time = frame_start
        if first_decay_cor is None:
            first_decay_cor = decay_cor

        # Remove original decay correction
        frame_data /= decay_cor
        
        # Scale by frame duration (convert minutes to seconds)
        frame_data *= (frame_duration * 60)
        
        # Sum the data
        summed_data += frame_data
        total_duration += frame_duration

    # Apply new decay correction
    if half_life > 0:
        if use_first_decay:
            decay_cor = first_decay_cor
        else:
            mid_time = start_time + total_duration / 2  # in minutes
            decay_cor = decay_correction(mid_time, start_time, half_life)
        summed_data *= decay_cor

    # Normalize by total duration (convert minutes to seconds)
    summed_data /= (total_duration * 60)

    # Apply final scale factor
    summed_data *= scale_factor

    # Create a new NIFTI image with the summed data
    summed_img = nib.Nifti1Image(summed_data, img.affine, img.header)

    # Save the summed image
    try:
        nib.save(summed_img, output_file)
    except IOError:
        raise IOError(f"Could not save PET sum file {output_file}")
    
    return output_file

#def parse_list_arg(arg):
#    return [float(x) for x in arg.split(',')]

def main():
    parser = argparse.ArgumentParser(description="Sum PET NIFTI images with decay correction.")
    parser.add_argument("--input_file", type=str, help="Input NIFTI file")
    parser.add_argument("--start_frame", type=int, help="Starting frame")
    parser.add_argument("--end_frame", type=int, help="Ending frame")
    parser.add_argument("--half-life", type=float, required=True, help="Half-life in seconds")
    parser.add_argument("--start-times", type=list, required=True, help="Comma-separated list of start times (in minutes)")
    parser.add_argument("--durations", type=list, required=True, help="Comma-separated list of durations (in minutes)")
    parser.add_argument("--decay-corrections", type=list, required=True, help="Comma-separated list of decay corrections")
    parser.add_argument("--scale", type=float, default=1.0, help="Final scale factor")
    parser.add_argument("--use-first-decay", action="store_true", help="Use decay correction of the first frame")
    parser.add_argument("--output_dir", type=str, default=None, required=False, help="Output directory")

    args = parser.parse_args()

    output_file = model_sum_pet(
        args.input_file, 
        args.start_frame, 
        args.end_frame, 
        args.half_life,
        args.start_times,
        args.durations,
        args.decay_corrections,
        args.scale, 
        args.use_first_decay,
        args.output_dir
    )

    print(f"Model summed file: {output_file}")

if __name__ == "__main__":
    main()