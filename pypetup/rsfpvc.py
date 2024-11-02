import argparse
import os

import numpy as np
import pandas as pd

from .misc import write_dataframe_to_csv
from .suvr import calculate_suvr, calculate_suvrlr, extract_roi_data

'''
def rsfpvc(rsfmat, roimean, iters):
    """
    rsfmat: a dictionary containing 'n' (matrix size) and 'mat' (n x n numpy array)
    roimean: a numpy array of size n
    iters: the number of iterations to be performed
    Returns a numpy array of size n
    """
    NROI = rsfmat.shape[0]
    r = np.zeros(NROI)
    val = np.zeros(NROI)
    m = np.copy(roimean)  # Initial estimation
    
    for k in range(iters):
        for i in range(NROI):
            val[i] = 0.
            for j in range(NROI):
                val[i] += m[j] * rsfmat[j, i]  # Blurred regional value assuming current estimation
            r[i] = abs(roimean[i] / val[i])  # ratio between reblurred value and observed value
            r[i] = min(max(r[i], 0.8), 1.2)  # constraints for the ratio to avoid blow up
            
        for i in range(NROI):
            m[i] *= r[i]  # Apply correction for current iteration

    return m
'''

def rsfpvc(rsfmat, roimean, iters=8):
    """
    Performs the actual RSF PVC correction
    Args:
        rsfmat (numpy.ndarray): A numpy array of shape (n, n)
        roimean (numpy.ndarray): A numpy array of size n
        iters (int, optional): Number of iterations to be performed. Default = 8.
    Returns:
         m (numpy.ndarray): Returns the RSF corrected mean values.
    """

    m = np.copy(roimean)  # Initial estimation

    steps = []  # To store intermediate results for each iteration

    for k in range(iters):
        val = rsfmat @ m  # Matrix-vector multiplication for blurred regional value
        r = np.abs(roimean / val)  # Ratio between reblurred value and observed value
        r = np.clip(r, 0.8, 1.2)  # Constraints for the ratio to avoid blow up
        m *= r  # Apply correction for current iteration
        steps.append(np.copy(m))  # Store the current state of m

    return m

def apply_rsfpvc(pet_file, rsfmat_file=None, rsfmask=None, iters=8):
    """
    Performs RSF PCV correction

    Args:
        pet_file (str): Path to the PET file.
        rsfmat_file (str, optional): Path to the rsfmat file. Defaults to None.
        rsfmask (str, optional): Path to the RSF Label image. Defaults to None.
        iters (int, optional): Number of iterations for computing RSF PVC. Defaults to 8.

    Raises:
        IOError: Error loading rsfmat.txt file.

    Returns:
        None
    """

    if rsfmat_file is None:
        rsfmat_file = os.path.join(os.path.dirname(pet_file), "rsfmat.txt")

    # Read RSFMat.txt
    try:
        rsfmat = np.loadtxt(rsfmat_file, delimiter="\t")
    except Exception as e:
        raise IOError(f"Error reading {rsfmat_file}: {e}")

    if rsfmask is None:
        rsfmask = os.path.join(os.path.dirname(pet_file), "RSFMask.nii.gz")

    df = extract_roi_data(rsfmask, pet_file)
    # get the roi mean values into a numpy column vector
    roimean = df["Mean_Signal"].to_numpy().reshape(-1, 1)
    #roimean = df["Mean_Signal"].to_numpy().reshape(1, -1)
    # Run the RSF PVC algorithm
    rsf_mean = rsfpvc(rsfmat, roimean, iters)
    rsf_df = df.drop('Mean_Signal', axis=1)
    rsf_df["Mean_Signal"] = rsf_mean

    # Compute SUVRLR
    suvrlr, ref_value = calculate_suvrlr(rsf_df)
    # Compute SUVR
    suvr = calculate_suvr(rsf_df, ref_value)

    suvrlr_file = os.path.join(os.path.dirname(pet_file), "RSF_SUVRLR.csv")
    suvr_file = os.path.join(os.path.dirname(pet_file), "RSF_SUVR.csv")

    # Write output files
    write_dataframe_to_csv(suvrlr, suvrlr_file)
    write_dataframe_to_csv(suvr, suvr_file)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run RSF PVC algorithm.")
    parser.add_argument(
        "--pet_file", type=str, required=True, help="Path to the msum pet file."
    )
    parser.add_argument(
        "--rsfmat",
        type=str,
        default=None,
        required=False,
        help="Path to the RSF matrix (CSV file).",
    )
    parser.add_argument(
        "--rsfmask",
        type=str,
        default=None,
        required=False,
        help="Path to the ROI mean vector (CSV file).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=8,
        required=False,
        help="Number of iterations. Default is 8.",
    )

    args = parser.parse_args()

    apply_rsfpvc(args.pet_file, args.rsfmat, args.rsfmask, args.iters)


if __name__ == "__main__":
    main()
