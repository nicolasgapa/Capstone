"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Main program (GUI).

"""
# Imports.
from data_processing import energy_spectrum
from math import ceil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from neural_networks import network_4, network_5, network_7
import numpy as np
import pandas as pd
from signal_processing import butterworth, detrend

# Inputs.
file = "data\\training\\107309.csv"
neural_networks = [network_4(6, 9910), network_5(6, 9910), network_7(6, 9910)]
weights = ['data\\weights\\weights_4.h5', 'data\\weights\\weights_5.h5', 'data\\weights\\weights_7.h5']
signal_processing_interval = 4
average_confidence = 5
exp_power = 8
max_window_size = 40
start_time = 100
confidence_threshold = 0.95

# Program.
df = pd.read_csv(file, header=None)
labels_file = 'data\\trainingAnswers.csv'
labels = pd.read_csv(labels_file)
label, peak = labels[labels['RunID'] == int(file[-10:-4])].to_numpy()[0][1:]
sources = {1: 'Uranium', 2: 'Plutonium', 3: 'Iodine', 4: 'Cobalt', 5: 'Technetium', 6: 'HEU + Tech'}
print('Source: {}. Peak at {}s.'.format(label, peak))

# Compute the actual time of each row (in seconds), and add the times column to the dataframe
time_sum, times = 0, []
for row in df[0]:
    time_sum += row
    times.append(time_sum / 1e6)
df['time'] = times
end_time = max(times)
# end_time = 180

# Create plots.
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(4, 6)
ax1 = fig.add_subplot(gs[:4, 0:2])
ax2 = fig.add_subplot(gs[:2, 2:4])
ax2.set_ylabel('Confidence')
ax2.plot([start_time, end_time], [confidence_threshold**exp_power, confidence_threshold**exp_power])
ax3 = fig.add_subplot(gs[2:4, 2:4])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Energy deviation from the mean (kEv)')
ax4 = fig.add_subplot(gs[0:4, 4:6])
ax4.text(0.35, 0.95, 'Summary:')
ax4.axis('off')

# Load networks.
loaded_networks = []
for network, weight in zip(neural_networks, weights):
    network.load_weights(weight)
    loaded_networks.append(network)

# Process.
ts, means, cs = [], [], []
xs, ys, signal_mean = [], [], 0
summary_ct, max_ys = 1, 0
for e, t in enumerate(range(start_time, ceil(end_time), 1), 0):

    # Update car position.
    progress_percentage = (t - start_time) / (end_time - start_time)
    ax1.clear()
    ax1.imshow(mpimg.imread('data\\city_model.png'), origin='lower')
    y_loc = (peak - start_time)*850 / (end_time - start_time)
    ax1.imshow(mpimg.imread('data\\rad_warning.png'), extent=(60, 100, y_loc, y_loc + 40), origin='lower')
    ax1.plot(108, 850 * progress_percentage, 'o', color='k')
    ax1.axis('off')

    # Date to process.
    df1 = df[df['time'] < t]
    df1 = df1[df1['time'] >= t - 10]

    # Process last few seconds, and generate the input to the neural network (spectra).
    spectra, spectra_2, spectra_3 = [], [], []
    for i in range(1, 11):
        # Obtain bins and spectrum.
        bins, spectrum = energy_spectrum(df1.copy(), t, i, n=10, kev_bin_size=1, max_kev=1000, plot=False)
        bins_2, spectrum_2 = energy_spectrum(df1.copy(), t, i, n=10, kev_bin_size=2, max_kev=2000, plot=False)
        bins_3, spectrum_3 = energy_spectrum(df1.copy(), t, i, n=10, kev_bin_size=3, max_kev=3000, plot=False)

        # Detrend.
        spectrum = detrend(bins, spectrum, 3)
        spectrum_2 = detrend(bins_2, spectrum_2, 3)
        spectrum_3 = detrend(bins_3, spectrum_3, 3)

        # Append.
        spectra.extend(spectrum)
        spectra_2.extend(spectrum_2)
        spectra_3.extend(spectrum_3)

    # Predict, and find the average of the network predictions.
    predictions = []
    for network, s in zip(loaded_networks, [spectra, spectra_2]):
        predictions.append(network.predict(np.array([s]))[0])
    predictions = list(pd.DataFrame(predictions).mean(axis=0))

    # Obtain source and confidence.
    source = np.argmax(predictions) + 1
    confidence = max(predictions)

    # Compute the mean of the last few confidences.
    ts.append(t)
    cs.append(confidence)
    if len(cs) > 0:
        mean = (sum(cs[-average_confidence:]) / min(len(cs), average_confidence)) ** exp_power
    else:
        mean = confidence ** exp_power
    means.append(mean)

    # Print summary.
    if mean >= confidence_threshold**exp_power:
        ax4.text(0.1, 0.9 - 0.035 * summary_ct,
                 'Source detected at time {}s.: {}.'.format(round(t, 1), sources[source]), size=9)
        summary_ct += 1
        if summary_ct * 0.035 >= 0.9:
            ax4.clear()
            ax4.axis('off')
            ax4.text(0.35, 0.95, 'Summary:')

    # Plot.
    ax2.plot(ts, means, color='blue', linewidth=0.8)
    ax2.plot(t, confidence ** exp_power, '1', color='black', markersize=2)
    ax2.set_ylim([0, 1])
    if t - start_time > max_window_size:
        ax2.set_xlim([t - max_window_size, t])
    else:
        ax2.set_xlim([start_time, t])
    y_labels = [0, 0.8, 0.9, 0.95, 1]
    ax2.set_yticks([i ** exp_power for i in y_labels])
    ax2.set_yticklabels([str(round(s * 100, 0)) + ' %' for s in y_labels])

    # Signal processing method.
    if t % signal_processing_interval == 0:
        # Extract the data form the last interval.
        df2 = df1[df1['time'] >= t - signal_processing_interval]

        # Convert the data to an evenly-spaced time series (signal) format.
        # df2 = convert_to_signal(df2.copy())
        df2 = df2.rename(columns={1: 'energy'})

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
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Energy deviation from the mean (kEv)')
        local_max_ys = max(ys)
        if local_max_ys > max_ys:
            max_ys = local_max_ys
            ax4.text(0.1, 0.9 - 0.035 * summary_ct,
                     'Maximum energy dev found at time {}s: {} kEv.'.format(round(t, 2), round(max_ys, 1)),
                     size=9)
            if summary_ct * 0.035 >= 0.9:
                ax4.clear()
                ax4.axis('off')
                ax4.text(0.35, 0.95, 'Summary:')
            summary_ct += 1
        ax3.set_ylim([0, max(200, local_max_ys)])

        # Reset the mean every 30 seconds.
        if t % 60 == 0:
            signal_mean = 0

    # Remove the already processed data.
    if (t - start_time) >= max_window_size:
        ts, cs, means = ts[1:], cs[1:], means[1:]
        xs, ys = xs[1:], ys[1:]
        ax2.set_xlim([t - max_window_size, t])
        ax3.set_xlim([t - max_window_size, t])
    df = df[df['time'] >= t - 10]

    # Display.
    plt.pause(0.05)
plt.pause(40)
