import nibabel as nib
import numpy as np
import pandas as pd
import argparse
import sys
import os
from .FreeSurferColorLUT import FreeSurferColorLUT, ROIs


def extract_roi_data(label_file, pet_image_file):
    """
    Extract ROI data from the label and PET image files.

    Parameters:
    label_file (str): Path to the label NIfTI file.
    pet_image_file (str): Path to the PET image NIfTI file.
    label_dict (dict): Dictionary containing label IDs and corresponding label names.

    Returns:
    pd.DataFrame: DataFrame containing ROI Label ID, Label Name, Mean PET Value, and Number of Voxels.
    """
    try:
        # Load images using nibabel
        label_img = nib.load(label_file)
        pet_img = nib.load(pet_image_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading NIfTI files: {e}")
        sys.exit(1)

    # Get the data as numpy arrays
    label_data = label_img.get_fdata()
    pet_data = pet_img.get_fdata()

    # Get unique label IDs
    label_ids = np.unique(label_data)

    # Initialize lists to store the results
    roi_label_id = []
    label_names = []
    mean_values = []
    num_voxels = []
    label_dict = FreeSurferColorLUT

    # Iterate through each label and calculate mean PET value and number of voxels
    for label in label_ids:
        label = int(label)
        if label != 0:  # Assuming 0 is the background, skip it
            # Get a mask of the current label
            mask = (label_data == label)
            
            # Extract the values from the PET image corresponding to this label
            pet_values = pet_data[mask]
            
            # Calculate mean and number of voxels
            mean_value = np.mean(pet_values)
            voxel_count = np.sum(mask)
            
            # Append to the lists
            roi_label_id.append(label)
            mean_values.append(mean_value)
            num_voxels.append(voxel_count)
            
            # Get label name from the dictionary, use 'Unknown' if not found
            label_names.append(label_dict.get(str(label), 'Unknown'))

    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        'ROI Label ID': roi_label_id,
        'Label Name': label_names,
        'Mean Value': mean_values,
        'NVoxels': num_voxels
    })
    
    return df


def calculate_suvr(df, output_dir):
    """
    Calculate SUVR for ROIs in a PET image and save the result.

    Parameters:
    df (pd.DataFrame): DataFrame containing ROI data.
    output_dir (str): Path of direcotry to save the output CSV file.
    Returns:
        ref_value (float): reference value of the total cerebellum.
    """
    roi_name_1 = 'Left-Cerebellum-Cortex'
    roi_name_2 = 'Right-Cerebellum-Cortex'
    output_filename = 'SUVRLR.csv'
    output_file = os.path.join(output_dir, output_filename)
    
    # Get ROI Label IDs from the names
    try:
        roi_label_1 = df[df['Label Name'] == roi_name_1]['ROI Label ID'].values[0]
        roi_label_2 = df[df['Label Name'] == roi_name_2]['ROI Label ID'].values[0]
    except IndexError:
        print(f"Error: One or both ROI names ({roi_name_1}, {roi_name_2}) were not found in the label dictionary.")
        sys.exit(1)

    # Extract the rows corresponding to these ROI labels
    roi_1_row = df[df['ROI Label ID'] == roi_label_1]
    roi_2_row = df[df['ROI Label ID'] == roi_label_2]

    # Extract mean values and voxel counts for these two labels
    mean_1 = roi_1_row['Mean Value'].values[0]
    mean_2 = roi_2_row['Mean Value'].values[0]
    voxels_1 = roi_1_row['NVoxels'].values[0]
    voxels_2 = roi_2_row['NVoxels'].values[0]

    # Compute weighted mean
    ref_value = (mean_1 * voxels_1 + mean_2 * voxels_2) / (voxels_1 + voxels_2)

    # Create a new column 'SUVR' by dividing 'Mean PET Value' by the weighted mean
    df['SUVR'] = df['Mean Value'] / ref_value

    # Save the updated DataFrame to a CSV file if needed
    try:
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    return ref_value

def compute_weighted_mean_for_roi_list(df, ref_value, output_dir):
    """
    Compute the weighted mean for a list of ROIs and calculate SUVR using a provided weighted mean.

    Parameters:
    df (pd.DataFrame): DataFrame containing ROI data.
    ref_value (float): Weighted average value for cerebellum reference region.
    output_dir (str): Path of the directory to store the csv file.

    Returns:
    pd.DataFrame: DataFrame containing the label names from the list, weighted average, number of voxels, and SUVRs.
    """
    output_filename = 'SUVR.csv'
    output_file = os.path.join(output_dir, output_filename)
    roi_names_list = ROIs

    if not isinstance(roi_names_list, list):
        raise TypeError("roi_names_list should be a list of ROI names.")

    # Initialize lists to store the results
    combined_label_names = []
    combined_weighted_means = []
    combined_num_voxels = []
    combined_suvrs = []

    # Iterate through each ROI name in the list
    for roi_name in roi_names_list:
        # Filter ROI data based on matching strings in Label Name
        matching_rows = df[df['Label Name'].str.contains(roi_name, case=False, na=False)]
        
        if matching_rows.empty:
            print(f"Warning: No matching ROIs found for the provided ROI name: {roi_name}")
            total_voxels = 0
            weighted_mean = 0
            suvr = 0
        else:
            # Calculate weighted mean and total voxels for the matching rows
            total_voxels = matching_rows['NVoxels'].sum()
            weighted_mean = (matching_rows['Mean Value'] * matching_rows['NVoxels']).sum() / total_voxels

            # Calculate SUVR for the combined ROI
            suvr = weighted_mean / ref_value

        # Append to the lists
        combined_label_names.append(roi_name)
        combined_weighted_means.append(weighted_mean)
        combined_num_voxels.append(total_voxels)
        combined_suvrs.append(suvr)

    # Create a new DataFrame for the results
    result_df = pd.DataFrame({
        'Label Name': combined_label_names,
        'Mean Value': combined_weighted_means,
        'NVoxels': combined_num_voxels,
        'SUVR': combined_suvrs
    })

    # Save the updated DataFrame to a CSV file if needed
    try:
        result_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)

    return None

def report_suvr(label_file, pet_image_file, output_dir):
    """
    Main function to extract ROI data and calculate SUVR.

    Args:
        label_file (str): Path to the label NIfTI file.
        pet_image_file (str): Path to the PET image NIfTI file.
        output_file (str): Path to save the output CSV file.

    Returns:
        None
    """
    df = extract_roi_data(label_file, pet_image_file)
    ref_value = calculate_suvr(df, output_dir)
    
    # Compute weighted means for ROI list
    compute_weighted_mean_for_roi_list(df, ref_value, output_dir)

def main():
    """
    Main function that parses the input arguments and call the reporting function.
    """
    parser = argparse.ArgumentParser(description="Calculate SUVR for ROIs in a PET image based on label files.")
    parser.add_argument('label_file', type=str, help="Path to the label NIfTI file.")
    parser.add_argument('pet_image_file', type=str, help="Path to the PET image NIfTI file.")
    parser.add_argument('output_dir', type=str, help="Path of the directory to save the output CSV files.")
    args = parser.parse_args()
    report_suvr(args.label_file, args.pet_image_file, args.output_dir)

if __name__ == "__main__":
    main()
