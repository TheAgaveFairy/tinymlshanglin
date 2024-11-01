#!/usr/bin/python3

import pandas as pd
import numpy as np
import os

output_dir = '../threeclass_indices'
data_indices_dir = '../data_indices/'
files = os.listdir(data_indices_dir)

wanted_classes = {
    'SR':0,
    'VT':1
}

def get_label(wanted):
    return wanted_classes.get(wanted, 2) # default 2 for all other categories not in wanted_classes

def main():
    for fi in files:
        file_path = os.path.join(data_indices_dir, fi)
        print("opening", file_path)
        
        df = pd.read_csv(file_path) 
        df['label'] = df['Filename'].str.split('-').str[1].map(get_label)

        output_path = os.path.join(output_dir, fi)
        df.to_csv(output_path, index=None)
        print(output_path, "saved!")

if __name__ == '__main__':
    main()
    exit()
