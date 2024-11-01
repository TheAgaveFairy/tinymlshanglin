#!/usr/bin/python3

import numpy as np
import os

data_path = r'../../tinyml_contest_data_training/'
data_files = os.listdir(data_path)

def fftSparseGen(mask, trunc = False, inverse = False):
    limit = 1
    for fi in data_files:
        temp = np.loadtxt(os.path.join(data_path,fi))
        fft = np.fft.rfft(temp)
        
        if trunc:
            done = fft[:][mask]
        else: # zeroes otherwise
            done = np.where(mask, fft, 0) 
        #print(len(done))
        if inverse:
            done = np.fft.irfft(done)
        """
        if limit > 0:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,2))
            ax1.plot(temp)
            ax2.plot(done)
            plt.show()
            limit -= 1
        """
        if limit > 0:
            print("length of a file:", len(done))
            limit -= 1
        outName = os.path.join(r'fft_data', fi)
        #print(outName)
        np.savetxt(outName, done.real, fmt='%.7f')
    
    print("DONE")

def main():
    bandpass_mask = np.zeros(626, dtype=bool)
    pass_band_filter = list(range(15*5,55*5))
    bandpass_mask[pass_band_filter] = True

    fftSparseGen(bandpass_mask, trunc = True)

if __name__ == '__main__':
    main()
    exit()
