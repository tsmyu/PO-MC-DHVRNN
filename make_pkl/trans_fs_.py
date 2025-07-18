import os
import pandas as pd
import argparse
import glob
import numpy as np

# Specify the folder path
parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, required=True)
args, _ = parser.parse_known_args()
folder_path = args.folder_path

# Get a list of all CSV files in the folder
csv_files = glob.glob(f"{folder_path}/Env5*.csv")
print(csv_files)


# Function to interpolate data points for upsampling
def interpolate_500Hz(data, original_times):
    interpolated_data = []
    new_times = []
    for i in range(1, len(original_times)):
        # original time in seconds
        t0 = original_times[i - 1].total_seconds()
        t1 = original_times[i].total_seconds()
        # data points at t0 and t1
        y0 = data.iloc[i - 1]
        y1 = data.iloc[i]
        # number of intervals between t0 and t1 at 500Hz
        num_intervals = int((t1 - t0) * 500)
        for j in range(num_intervals):
            tj = t0 + j / 500.0
            interpolated_data.append(y0 + (y1 - y0) * (tj - t0) / (t1 - t0))
            new_times.append(pd.Timedelta(seconds=tj))
    # Append the last data point
    interpolated_data.append(data.iloc[-1])
    new_times.append(original_times[-1])
    return pd.DataFrame(interpolated_data, index=new_times)


# Iterate over each CSV file
for file in csv_files:
    print(f"target:{os.path.basename(file)}")
    data = pd.read_csv(file)
    # Save the original "Frame" column
    # original_frame_start = data["Frame"].iloc[0]
    # Convert the "Time (Seconds)" column to a timedelta for easy manipulation
    data["Time (Seconds)"] = pd.to_timedelta(data["Time (Seconds)"], unit="s")
    # Set the 'Time (Seconds)' column as the index
    data.set_index("Time (Seconds)", inplace=True)
    # Interpolate data to 500Hz
    original_times = data.index
    data_resampled_500Hz = interpolate_500Hz(data, original_times)
    # Downsample the data from 500Hz to 100Hz (0.01 seconds)
    new_index_100Hz = data_resampled_500Hz.index[
        ::5
    ]  # Select every 5th index to get 100Hz from 500Hz
    data_resampled_100Hz = data_resampled_500Hz.loc[new_index_100Hz]
    # Create new time index for 100Hz
    new_time_index_100Hz = pd.timedelta_range(
        start=data_resampled_100Hz.index.min(),
        periods=len(data_resampled_100Hz),
        freq="10L",
    )
    data_resampled_100Hz.index = new_time_index_100Hz
    # Convert the index back to the same format as the original 'Time (Seconds)' column
    data_resampled_100Hz.index = data_resampled_100Hz.index.total_seconds()
    # Reset index to bring 'Time (Seconds)' back as a column
    data_resampled_reset = data_resampled_100Hz.reset_index()
    data_resampled_reset.rename(
        columns={"index": "Time (Seconds)"}, inplace=True
    )
    # Add the "Frame" column with sequential values starting from the original first frame value
    # data_resampled_reset["Frame"] = np.arange(
    #     original_frame_start, original_frame_start + len(data_resampled_reset)
    # )
    # Save the resampled data to a new CSV file
    resample_folder = os.path.join(folder_path, "resample")
    os.makedirs(resample_folder, exist_ok=True)
    file_name = os.path.basename(file)
    resampled_file_path = os.path.join(
        resample_folder, file_name + "_resample.csv"
    )
    data_resampled_reset.to_csv(resampled_file_path, index=False)
