"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Main program (GUI).

"""
# Imports.
from data_processing import energy_spectrum
import numpy as np
import pandas as pd

# Inputs.
file = "data\\training\\104902.csv"


# Program.
csvfile = pd.read_csv(file, header=None)
labels_file = 'data\\trainingAnswers.csv'
labels = pd.read_csv(labels_file)
label, peak = labels[labels['RunID'] == int(file[-10:-4])].to_numpy()[0][1:]

xs = []
for i in range(1, 11):
    x, y = energy_spectrum(csvfile, peak, i, n=10, kev_bin_size=5, plot=False)
    xs.extend(x)
np.save('data\\npy\\' + file[-10:-4]+'.npy', np.array(xs))
