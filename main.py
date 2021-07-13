"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Main program (GUI).

"""
# Imports.
from data_processing import energy_spectrum
from neural_networks import network_5, network_7, network_9
import pandas as pd
from signal_processing import detrend

# Inputs.
file = "data\\training\\104903.csv"
neural_networks = [network_5(6, 9910), network_7(6, 9910), network_9(6, 9910)]
weights = ['data\\weights\\network_5.h5', 'data\\weights\\network_7.h5', 'data\\weights\\network_9.h5']

# Program.
csvfile = pd.read_csv(file, header=None)
labels_file = 'data\\trainingAnswers.csv'
labels = pd.read_csv(labels_file)
label, peak = labels[labels['RunID'] == int(file[-10:-4])].to_numpy()[0][1:]

# Process last few seconds, and generate the input to the neural network (spectra).
spectra = []
for i in range(1, 11):
    bins, spectrum = energy_spectrum(csvfile, peak, i, n=10, kev_bin_size=1, max_kev=1000, plot=False)
    spectrum = detrend(bins, spectrum, 3)
    spectra.extend(spectrum)
    print(len(spectra))

# Load neural networks and weights, and predict.
# predictions = []
# for network, weight in zip(neural_networks, weights):
#    network.load_weights(weight)
#    predictions.append(network.predict(spectra))

# Va en el 101355.
