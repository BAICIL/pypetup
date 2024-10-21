import nibabel as nib
import numpy as np
import pandas as pd
import argparse
import sys
import os
from .FreeSurferColorLUT import FreeSurferColorLUT, ROIs


def extract_roi_data(label_file, pet_image_file, label_dict):
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
            label_names.append(label_dict.get(int(label), 'Unknown'))

    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        'ROI Label ID': roi_label_id,
        'Label Name': label_names,
        'Mean PET Value': mean_values,
        'Number of Voxels': num_voxels
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
    mean_1 = roi_1_row['Mean PET Value'].values[0]
    mean_2 = roi_2_row['Mean PET Value'].values[0]
    voxels_1 = roi_1_row['Number of Voxels'].values[0]
    voxels_2 = roi_2_row['Number of Voxels'].values[0]

    # Compute weighted mean
    ref_value = (mean_1 * voxels_1 + mean_2 * voxels_2) / (voxels_1 + voxels_2)

    # Create a new column 'SUVR' by dividing 'Mean PET Value' by the weighted mean
    df['SUVR'] = df['Mean PET Value'] / ref_value

    # Save the updated DataFrame to a CSV file if needed
    try:
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    return ref_value

def compute_weighted_mean_for_roi_list(df, ref_value):
    """
    Compute the weighted mean for a list of ROIs and calculate SUVR using a provided weighted mean.

    Args:
        df (pd.DataFrame): DataFrame containing ROI data.
        ref_value (float): The weighted mean value to use for SUVR calculation.
        output_dir (str): Path of direcotry to save the output CSV file.

    Returns:
        None
    """
    output_filename = 'SUVRLR.csv'
    output_file = os.path.join(output_dir, output_filename)
    roi_names_list = ROIs
    # Filter ROI data based on matching strings in Label Name
    roi_data = df[df['Label Name'].str.contains('|'.join(roi_names_list), case=False, na=False)]

    # Calculate weighted mean for each ROI in the list
    weighted_means = []
    total_voxels = 0
    for roi_name in roi_names_list:
        matching_rows = roi_data[roi_data['Label Name'].str.contains(roi_name, case=False, na=False)]
        for _, row in matching_rows.iterrows():
            mean_value = row['Mean PET Value']
            voxel_count = row['Number of Voxels']
            weighted_means.append(mean_value * voxel_count)
            total_voxels += voxel_count
    
    # Compute SUVR for each ROI using the provided overall weighted mean
    roi_data['SUVR'] = roi_data['Mean PET Value'] / ref_value

    # Create a new DataFrame for the results
    result_df = roi_data[['Label Name', 'Mean PET Value', 'Number of Voxels', 'SUVR']]
    result_df.rename(columns={'Mean PET Value': 'Weighted Average'}, inplace=True)
    
    # Save the updated DataFrame to a CSV file if needed
    try:
        result_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
        
    return None

def report_suvr(label_file, pet_image_file, output_file):
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
    result_df = compute_weighted_mean_for_roi_list(df, ref_value, output_dir)

def main():
    """
    Main function that parses the input arguments and call the reporting function.
    """
    parser = argparse.ArgumentParser(description="Calculate SUVR for ROIs in a PET image based on label files.")
    parser.add_argument('label_file', type=str, help="Path to the label NIfTI file.")
    parser.add_argument('pet_image_file', type=str, help="Path to the PET image NIfTI file.")
    parser.add_argument('output_dir', type=str, help="Path of the directory to save the output CSV files.")

    report_suvr(args.label_file, args.pet_image_file, args.output_dir)

    
    

if __name__ == "__main__":
    main()
