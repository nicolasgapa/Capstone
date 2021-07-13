"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Main program (GUI).

"""
# Imports.
from data_processing import energy_spectrum
from math import ceil
import matplotlib.pyplot as plt
from neural_networks import network_4, network_5
import numpy as np
import pandas as pd
from signal_processing import butterworth, convert_to_signal, detrend

# Inputs.
file = "data\\training\\104902.csv"
neural_networks = [network_4(6, 9910), network_5(6, 9910)]  # , network_9(6, 9910)]
weights = ['data\\weights\\weights_4.h5', 'data\\weights\\weights_5.h5']  # , 'data\\weights\\network_9.h5']
signal_processing_interval = 5
average_confidence = 5
exp_power = 10

# Program.
df = pd.read_csv(file, header=None)
labels_file = 'data\\trainingAnswers.csv'
labels = pd.read_csv(labels_file)
label, peak = labels[labels['RunID'] == int(file[-10:-4])].to_numpy()[0][1:]
print('Source: {}. Peak at {}s.'.format(label, peak))

# Compute the actual time of each row (in seconds), and add the times column to the dataframe
time_sum, times = 0, []
for row in df[0]:
    time_sum += row
    times.append(time_sum / 1e6)
df['time'] = times
start_time = 7
end_time = max(times)

# Create plots.
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(4, 6)
ax2 = fig.add_subplot(gs[:2, 2:4])
ax3 = fig.add_subplot(gs[2:4, 2:4])

# Load networks.
loaded_networks = []
for network, weight in zip(neural_networks, weights):
    network.load_weights(weight)
    loaded_networks.append(network)

#
ts, means, cs = [], [], []
xs, ys, signal_mean = [], [], 0
for e, t in enumerate(range(start_time, ceil(end_time), 1), 0):
    # Date to process.
    df1 = df[df['time'] < t]
    df1 = df1[df1['time'] >= t - 10]

    # Process last few seconds, and generate the input to the neural network (spectra).
    spectra, spectra_2 = [], []
    for i in range(1, 11):
        # Obtain bins and spectrum.
        bins, spectrum = energy_spectrum(df1.copy(), t, i, n=10, kev_bin_size=1, max_kev=1000, plot=False)
        bins_2, spectrum_2 = energy_spectrum(df1.copy(), t, i, n=10, kev_bin_size=2, max_kev=2000, plot=False)

        # Detrend.
        spectrum = detrend(bins, spectrum, 3)
        spectrum_2 = detrend(bins_2, spectrum_2, 3)

        # Append.
        spectra.extend(spectrum)
        spectra_2.extend(spectrum_2)

    # Predict, and find the average of the network predictions.
    predictions = []
    for network, s in zip(loaded_networks, [spectra, spectra_2]):
        predictions.append(network.predict(np.array([s]))[0])
    predictions = list(pd.DataFrame(predictions).mean(axis=0))

    # Obtain source and confidence.
    source = np.argmax(predictions) + 1
    confidence = max(predictions)

    # Plot neural network results.
    ts.append(t)
    cs.append(confidence)
    if len(cs) > 0:
        means.append((sum(cs[-average_confidence:]) / min(len(cs), average_confidence)) ** exp_power)
    else:
        means.append(confidence ** exp_power)
    ax2.plot(ts, means, color='blue', linewidth=0.8)
    ax2.plot(t, confidence ** exp_power, '1', color='black', markersize=2)
    ax2.set_ylim([0, 1])
    y_labels = [0, 0.8, 0.9, 0.95, 1]
    ax2.set_yticks([i ** exp_power for i in y_labels])
    ax2.set_yticklabels([str(round(s*100, 0)) + ' %' for s in y_labels])

    # Signal processing method.
    if t % signal_processing_interval == 0:
        # Extract the data form the last interval.
        df2 = df1[df1['time'] >= t - signal_processing_interval]

        # Convert the data to an evenly-spaced time series (signal) format.
        df2 = convert_to_signal(df2.copy())

        # Apply a butterworth to the signal.
        x, y = butterworth(df2.copy(), 0, 1, n=100, filter_order=1, plot=False)

        # Get rid of the initial tail (typical of a butterworth filter output as it uses fourier series).
        x, y = x[100 * 3:], y[100 * 3:]
        if signal_mean == 0:
            signal_mean = np.mean(y)
        else:
            signal_mean = (signal_mean * e + np.mean(y)) / (e + 1)
        y = [abs(i - signal_mean) for i in y]
        xs.extend(x)
        ys.extend(y)

        # Plot.
        ax3.clear()
        ax3.plot(xs, ys, color='red', linewidth=0.4)
        ax3.set_ylim([0, max(80, max(ys))])

        # Reset the mean every minute.
        if t % 60 == 0:
            signal_mean = 0

    # Display.
    plt.pause(0.1)
plt.pause(30)
