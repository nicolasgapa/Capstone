"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Data processing.

"""
# Imports.
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from signal_processing import detrend


# Functions.
def energy_spectrum(df, peak_time, minutes_to_process, n=10, kev_bin_size=5, max_kev=3000, plot=True):
    """
    Produce and energy spectrum plot, and return the x, y vectors of such plot.

    :param df: Dataframe of 'time' vs 'energy'.
    :param peak_time: Event time.
    :param minutes_to_process: Minutes before the peak time to consider.
    :param n: Rolling mean window size.
    :param kev_bin_size: Size of the bins of the energy spectrum.
    :param max_kev: Maximum kEv value.
    :param plot: Plot the energy spectrum (True) or not (False).
    :return: x, y: Energy (in kEv) vs. frequency (log(counts)).
    """

    # Extract minutes before the peak time. Filter out the rest of the data.
    df = df[df['time'] >= peak_time - minutes_to_process]
    df = df[df['time'] <= peak_time]

    # Group by energy level.
    df['energy_level'] = [floor(t / kev_bin_size) * kev_bin_size for t in df[1]]
    df = df[df['energy_level'] < max_kev]
    df = df.groupby('energy_level').count().drop([1, 'time'], axis=1)
    full_idx = [i for i in range(0, max_kev, kev_bin_size)]
    df2 = pd.DataFrame([0] * len(full_idx), index=full_idx)
    df3 = df.add(df2, fill_value=0) / minutes_to_process

    # Apply a rolling mean and log.
    rolling_mean = df3.rolling(n).mean()
    log_rolling = np.log10(rolling_mean[n - 1:])
    log_rolling.loc[log_rolling[0] <= -2, 0] = -2
    x, y = log_rolling.index, log_rolling[0]

    # Plot.
    if plot:
        plt.plot(x, y)
        plt.show()

    # Return x and y.
    return x, y


def numpy_files(input_folder, output_folder, labels_file='data\\trainingAnswers.csv'):
    """
    Generate npy files to train the neural networks.

    :param input_folder: Files to process.
    :param output_folder: Directory to save numpy files.
    :param labels_file: File indicating the labels.
    :return: None (saves the files to the output folder).
    """
    # Process each file in the input folder.
    for file in os.listdir(input_folder):

        # Read file and labels.
        csvfile = pd.read_csv(input_folder + '\\' + file, header=None)
        csvfile = add_time_column(csvfile)
        labels = pd.read_csv(labels_file)
        label, peak = labels[labels['RunID'] == int(file[-10:-4])].to_numpy()[0][1:]

        # Simulations withot a source.
        if label == 0:
            peak = 30

        # Generate the enrgy spectrum.
        spectra = []
        for i in range(1, 11):
            bins, spectrum = energy_spectrum(csvfile, peak, i, n=10, kev_bin_size=3, max_kev=3000, plot=False)
            spectrum = detrend(bins, spectrum, 3)
            spectra.extend(spectrum)

        # Save numpy file.
        np.save(output_folder + '\\' + file[-10:-4] + '.npy', np.array(spectra))
        print('Saved: {}.'.format(file[-10:-4] + '.npy'), 'Peak: {} s.'.format(peak), 'Label: {}.'.format(label))


def add_time_column(df):
    """
    Takes the raw data and adds a column with the actual time ('time')

    :param df: Dataframe containing the original data.
    :return: Updated df.
    """

    # Compute the actual time of each row (in seconds).
    time_sum, times = 0, []
    for row in df[0]:
        time_sum += row
        times.append(time_sum / 1e6)

    # Add the times column to the dataframe.
    df['time'] = times

    # Return
    return df


# numpy_files('data\\training', 'data\\npy\\npy_3000')
