import sys
import os
import glob
import pandas as pd
import numpy as np


def main():
    raw_data_path = '/Users/johakim/Desktop/EPIC-Lab-Project/Raw-Data'
    os.chdir(raw_data_path)
    raw_data_files = glob.glob('*.csv')

    window_size = int(sys.argv[1])

    feature_extracted_path = '/Users/johakim/Desktop/EPIC-Lab-Project/Feature-Extracted/'
    extension = '.csv'

    for file in raw_data_files:
        raw_data = pd.read_csv(file)
        print('Raw data file: {}\n'.format(file), '-'*60)
        print('Raw data df shape: {}\n'.format(raw_data.shape), '-' * 60)

        # split raw data into sensor data and labels
        sensor_data = raw_data.iloc[:, :-2]
        labels = raw_data.iloc[:, -2:]
        print('Sensor data df columns:\n{}\n'.format(sensor_data.columns), '-'*60)
        print('Label df columns:\n{}\n'.format(labels.columns), '-'*60)

        # feature extraction
        feature_extracted_df = feature_extract(sensor_data, labels, window_size)
        print('Feature extracted df:\n{}\n'.format(feature_extracted_df), '-'*60)

        # save feature extracted dataframe
        output_file_path = feature_extracted_path + os.path.splitext(file)[0] + extension
        feature_extracted_df.to_csv(output_file_path, index=False)


def feature_extract(sensor_data, labels, window_size):
    # df contains sensor data
    orig_shape = sensor_data.shape
    output_df = pd.DataFrame()

    # loop over each column in df
    for i in range(orig_shape[1]):
        # generate sliding window as a new 2D array (row: a single sliding window, col: window_size index)
        # for all data collected by a single sensor, this generates a data frame with all possible windows
        single_column = sensor_data.iloc[:, i].values
        shape_des = single_column.shape[:-1] + (single_column.shape[-1] - window_size + 1, window_size)
        strides_des = single_column.strides + (single_column.strides[-1],)
        sliding_window = np.lib.stride_tricks.as_strided(single_column, shape=shape_des, strides=strides_des)

        # extract 6 features for a specific sensor column over every window
        sensor_name = sensor_data.columns[i]
        min = pd.Series(sliding_window.min(axis=1), name='Min '+sensor_name)
        max = pd.Series(sliding_window.max(axis=1), name='Max '+sensor_name)
        mean = pd.Series(sliding_window.mean(axis=1), name='Mean '+sensor_name)
        std = pd.Series(sliding_window.std(axis=1), name='Std '+sensor_name)
        first = pd.Series(sliding_window[:, 0], name='First '+sensor_name)
        last = pd.Series(sliding_window[:, -1], name='Last '+sensor_name)

        # append the 6 features into one 2D array
        sensor_features = pd.concat([min, max, mean, std, first, last], axis=1)
        output_df = pd.concat([output_df, sensor_features], axis=1)

    # create new label columns
    new_labels = labels.iloc[window_size - 1:, :].reset_index(drop=True)

    # return output data frame
    return pd.concat([output_df, new_labels], axis=1)


if __name__ == "__main__":
    main()