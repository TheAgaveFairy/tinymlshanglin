#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
from collections import defaultdict

balanced_indices_dir = '../balanced_indices/' # output
data_indices_dir = '../data_indices/'
files = os.listdir(data_indices_dir)

label_list = ['AFb','AFt','SR','SVT','VFb','VFt','VPD','VT']

labels = {}
for i, label in enumerate(label_list):
    labels[label] = i
print("labels:", labels)
def getLabel(oldLabel):
    return label_list.index(oldLabel)

chance_to_keep = 0.10
categories_to_cut = ['SR', 'VT']

def main():
    for idx_file in os.listdir(data_indices_dir):
        idx_path = os.path.join(data_indices_dir, idx_file)
        print("Processing:",idx_path)
        
        df = pd.read_csv(idx_path, sep=',', header = 0)
        
        ### Make into 8-way multiclass
        df['label'] = df['Filename'].str.split('-').str[1].map(getLabel)
        ###

        mask = df.Filename.str.split('-').str[1].isin(categories_to_cut)
        random_nums = np.random.random(len(df))                 # [0.0 - 1.0] * n
        keep_mask = ~(mask & (random_nums > chance_to_keep))    # get rid of 1-chance_to_keep rows
        df = df[keep_mask].reset_index(drop=True)

        out_path = os.path.join(balanced_indices_dir, idx_file)
        df.to_csv(out_path, index=None)

    vs = "Processing complete, " + str(chance_to_keep * 100) + "% of " + str(categories_to_cut) + " kept.\n"
    print(vs) # verification string to print
    print("RESULTS:")

    for idx_file in os.listdir(balanced_indices_dir):
        counts = defaultdict(int)
        idx_path = os.path.join(balanced_indices_dir, idx_file)
        df_indices = pd.read_csv(idx_path, sep=",", header=0)
        names = df_indices.Filename.to_list()
        
        fns = []
        labs = []
        for filename in names:
            parts = filename.split('-')
            #pn = parts[0][1:] #patient number
            label = parts[1]
            counts[label] += 1
        out_of = 24588
        if 'test' in idx_file:
            out_of = 5625
        vs = str(len(names)) + "/ " + str(out_of) + " files kept: " + str(counts) 
        print(vs)

if __name__ == '__main__':
    main()
    exit()
