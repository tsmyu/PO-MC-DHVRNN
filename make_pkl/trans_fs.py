import os
import pandas as pd
import argparse
import glob
import os
import pandas as pd
import argparse
import glob

# Specify the folder path
parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, required=True)
args, _ = parser.parse_known_args()
folder_path = args.folder_path

# Get a list of all CSV files in the folder
csv_files = glob.glob(f"{folder_path}/Env5/*/*.csv")
# csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
print(csv_files)

# Iterate over each CSV file
for file in csv_files:
    print(f"target:{os.path.basename(file)}")
    data = pd.read_csv(file)
    # Convert the "Time (Seconds)" column to a timedelta for easy resampling
    data["Time (Seconds)"] = pd.to_timedelta(data["Time (Seconds)"], unit="s")
    # Set the 'Time (Seconds)' column as the index
    data.set_index("Time (Seconds)", inplace=True)
    # Resample the data at 100Hz (every 0.01 seconds)
    data_resampled = data.resample("10L").interpolate(
        "time"
    )  # '10L' means 10 milliseconds
    # data_resampled = data.resample("10L").interpolate("time")
    # Convert the index back to the same format as the original 'Time (Seconds)' column
    data_resampled.index = data_resampled.index.total_seconds()
    # Reset index to bring 'Time (Seconds)' back as a column
    data_resampled_reset = data_resampled.reset_index()
    data_resampled_reset.rename(
        columns={"index": "Time (Seconds)"}, inplace=True
    )
    # Save the resampled data to a new CSV file
    resample_folder = os.path.join(folder_path, "resample")
    os.makedirs(resample_folder, exist_ok=True)
    file_name = os.path.basename(file)
    resampled_file_path = os.path.join(
        resample_folder, file_name + "_resample.csv"
    )
    data_resampled_reset.to_csv(resampled_file_path, index=False)
