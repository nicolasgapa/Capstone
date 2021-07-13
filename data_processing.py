"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Data processing.

"""
# Imports.
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Functions.
def energy_spectrum(df, peak_time, minutes_to_process, n=10, kev_bin_size=5, plot=True):
    """
    Produce and energy spectrum plot, and return the x, y vectors of such plot.

    :param df: Dataframe (original data).
    :param peak_time: Event time.
    :param minutes_to_process: Minutes before the peak time to consider.
    :param n: Rolling mean window size.
    :param kev_bin_size: Size of the bins of the energy spectrum.
    :param plot: Plot the energy spectrum (True) or not (False).
    :return: x, y: Energy (in kEv) vs. frequency (log(counts)).
    """

    # Compute the actual time of each row (in seconds).
    time_sum, times = 0, []
    for row in df[0]:
        time_sum += row
        times.append(time_sum / 1e6)

    # Add the times column to the dataframe.
    df['times'] = times

    # Extract seconds before the peak time. Filter out the rest of the data.
    df = df[df['times'] >= peak_time - minutes_to_process]
    df = df[df['times'] <= peak_time]

    # Group by energy level.
    df['energy_level'] = [floor(t / kev_bin_size) * kev_bin_size for t in df[1]]
    df = df[df['energy_level'] < 3000]
    df = df.groupby('energy_level').count().drop([1, 'times'], axis=1)
    full_idx = [i for i in range(0, 3000, kev_bin_size)]
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
