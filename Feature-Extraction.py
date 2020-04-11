import sys
import os
import glob
import pandas as pd
import numpy as np



def main():
    raw_data_path = '/Users/johakim/Desktop/EPIC-Lab-Project/Raw-Data'
    os.chdir(raw_data_path)
    raw_data_files = glob.glob('*.csv')

    # window_size = int(sys.argv[1])
    window_size = 30

    feature_extracted_path = '/Users/johakim/Desktop/EPIC-Lab-Project/Feature-Extracted'
    extension = '.csv'

    for file in raw_data_files:
        raw_data = pd.read_csv(file)
        print('Raw data file: {}\n'.format(file), '-'*60)

        # split raw data into sensor data and labels
        sensor_data = raw_data.iloc[:, :-2]
        labels = raw_data.iloc[:, -2:]
        print('Sensor data df columns:\n{}\n'.format(sensor_data.columns), '-'*60)
        print('Label df columns:\n{}\n'.format(labels.columns), '-'*60)




if __name__ == "__main__":
    main()