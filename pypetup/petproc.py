import argparse
import os

import pypetup as pup


def run_pup(
    pet_nifti,
    fs_dir,
    pet_json=None,
    derivatives_dir=None,
    t1_filename="orig_nu.mgz",
    start_time=None,
    duration=None,
    norsf=False,
):

    print("Copying PET data to derivatives directory")
    input_pet_dir = os.path.dirname(pet_nifti)
    input_pet_filename = os.path.basename(pet_nifti)
    input_pet_filename_without_extension = input_pet_filename.split(".")[0]

    if pet_json is None:
        pet_json = os.path.join(
            input_pet_dir, input_pet_filename_without_extension + ".json"
        )

    if derivatives_dir is None:
        process_folder = os.path.join(input_pet_dir, "derivatives")
        os.makedirs(process_folder, exist_ok=True)
    else:
        parts = input_pet_filename.split("_")
        subject_id = parts[0]
        session_id = parts[1]
        tracer = parts[2][7:]
        process_folder = os.path.join(derivatives_dir, subject_id, session_id, tracer)
        os.makedirs(process_folder, exist_ok=True)

    copy_pet_nifti = os.path.join(process_folder, input_pet_filename)
    copy_pet_json = os.path.join(
        process_folder, input_pet_filename_without_extension + ".json"
    )

    _ = pup.time_function(pup.copy_file, **{"src": pet_json, "dest": copy_pet_json})
    _ = pup.time_function(pup.copy_file, **{"src": pet_nifti, "dest": copy_pet_nifti})

    print("Performing motion correction")
    mocofile = pup.time_function(pup.perform_motion_correction, (copy_pet_nifti))

    print("Processing MRI FreeSurfer data")
    t1_mgz = os.path.join(fs_dir, t1_filename)
    wmparc_mgz = os.path.join(fs_dir, "wmparc.mgz")
    _ = pup.time_function(pup.convert_mgz_to_nii, *(t1_mgz, wmparc_mgz, process_folder))

    print("Co-registering PET to T1")
    t1file = os.path.join(process_folder, "T1.nii.gz")
    coregfile = pup.time_function(pup.coregister_pet_to_t1, *(mocofile, t1file))

    print("Get model frame indices")
    start_frame, end_frame = pup.find_frame_indices(copy_pet_json, start_time, duration)

    print("Generate msum image")
    data = pup.load_json(copy_pet_json)
    half_life = data["RadionuclideHalfLife"]
    start_times = data["FrameTimesStart"]
    durations = data["FrameDuration"]
    decay_factors = data["DecayFactor"]
    msumfile = pup.time_function(
        pup.model_sum_pet,
        *(
            coregfile,
            start_frame,
            end_frame,
            half_life,
            start_times,
            durations,
            decay_factors,
        )
    )

    print("Generate uncorrected SUVR tables")
    label_file = os.path.join(process_folder, "wmparc.nii.gz")
    _ = pup.time_function(pup.report_suvr, *(label_file, msumfile))

    if not norsf and data["Smoothed"] == "yes":
        print("Preparing to perform RSF correction")
        headmask_file = os.path.join(process_folder, "headmask.nii.gz")
        _ = pup.time_function(pup.prepare_for_rsf, (headmask_file))

        print("Performing RSF correction and generating corrected SUVR tables")
        _ = pup.time_function(pup.apply_rsfpvc, (msumfile))

    else:
        print("RSF correction not performed because:")
        print("1. It is not request by User")
        print("2. The input data was not harmonized to 8mm3 filter")

    return None


def main():
    """
    Main Function to handle command-line arguments and perform PET processing.

    Uses argparse to campture command-line inputs.
    """
    parser = argparse.ArgumentParser(
        description="Process PET data using the PUP workflow."
    )
    parser.add_argument(
        "--pet_nifti", type=str, required=True, help="Path to the input PET file."
    )
    parser.add_argument(
        "--pet_json",
        type=str,
        required=False,
        default=None,
        help="Path to the input PET JSON file.",
    )
    parser.add_argument(
        "--derivatives_dir",
        type=str,
        required=False,
        default=None,
        help="Path to the base directory (optonal, default=None).",
    )
    parser.add_argument(
        "--fs_dir",
        type=str,
        required=True,
        help="Path of the Subjects FreeSurfer mri directory.",
    )
    parser.add_argument(
        "--t1_filename",
        type=str,
        required=False,
        default="orig_nu.mgz",
        help="File name with extension of the T1 file (optional, default=orig_nu.mgz)",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        required=False,
        default=None,
        help="Start time for frames of interest (optional, default=None)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        required=False,
        default=None,
        help="Total duration of frames of interest (optional, default=None)",
    )
    parser.add_argument(
        "--norsf", action="store_true", help="Do not perform RSF correction"
    )

    args = parser.parse_args()

    result = pup.time_function(
        run_pup,
        *(
            args.pet_nifti,
            args.fs_dir,
            args.pet_json,
            args.derivatives_dir,
            args.t1_filename,
            args.start_time,
            args.duration,
            args.norsf,
        )
    )


if __name__ == "__main__":
    main()
