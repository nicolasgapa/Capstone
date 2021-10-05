"""

Embry-Riddle Aeronautical University
Author: Nicolas Gachancipa
Name: Signal processing methods.

"""

# Imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


# Functions.
def convert_to_signal(df):
    """
    Converts the original (unevenly spaced) data to an equally-spaced time series (a signal).

    :param df: Radiation data with two columns: time (in s) and energy (kEv).
    :return: two vectors: times (since the start of the simulation) and energy levels (in kEv).

    """
    # Average or round values.
    average = True
    if average:
        new_df = []
        for index, row in df.iterrows():
            raw = row['time']
            rounded = round(raw, 3)
            if raw - rounded < 0:
                up = 1 - abs(raw - rounded) * 1000
                down = 1 - up
                new_df.append([round(rounded - 0.001, 3), down * row[1]])
                new_df.append([round(rounded, 3), up * row[1]])
            else:
                down = 1 - abs(raw - rounded) * 1000
                up = 1 - down
                new_df.append([round(rounded, 3), down * row[1]])
                new_df.append([round(rounded + 0.001, 3), up * row[1]])

        df1 = pd.DataFrame(np.array(new_df)).astype(float)
        df1 = df1.groupby(0).mean()
    else:
        df['rounded_time'] = [round(t, 3) for t in df['time']]
        df1 = df.groupby('rounded_time').mean()

    # Build signal.
    times, radiation_levels = df1.index, df1[1]
    sig = pd.DataFrame({'time': times, 'energy': radiation_levels})

    # Return.
    return sig


def detrend(x_values, y_values, order, absolute=False):
    # Detrend.
    poly_coef = np.polyfit(x_values, y_values, order)  # Degree of the polynomial.
    poly = np.polyval(poly_coef, x_values)  # In this case, y-axis is the TEC vector.
    polyfit_tec = y_values - poly
    if absolute:
        polyfit_tec = abs(polyfit_tec)
    return polyfit_tec


# Low pass butterworth.
def detrending_low_pass(y_values, cutoff=10, order=5):
    """"
    Function: Detrend the data and use a butterworth filter.
    Inputs:
        x_values (list): time values.
        y_values (list): TEC values.
        poly_degree (int): Degree of the polynomial.
        cutoff (float): Desired cut off frequency [Hz]
        order (int): Order of the butterworth filter.
    Output:
        This function returns the detrended TEC (y-axis) values only.
    """

    # Filter
    low_butterworth = signal.butter(order, cutoff, 'lowpass', fs=1000, output='sos')
    detrended_tec = signal.sosfilt(low_butterworth, y_values)

    # Return the detrended TEC vector (y-axis values).
    return detrended_tec


# High pass butterworth.
def detrending_high_pass(y_values, cutoff=0.1, order=5):
    """"
    Function: Detrend the data and use a butterworth filter.
    Inputs:
        x_values (list): time values.
        y_values (list): TEC values.
        poly_degree (int): Degree of the polynomial.
        cutoff (float): Desired cut off frequency [Hz]
        order (int): Order of the butterworth filter.
    Output:
        This function returns the detrended TEC (y-axis) values only.
    """

    # Filter
    high_butterworth = signal.butter(order, cutoff, 'highpass', fs=1000, output='sos')
    detrended_tec = signal.sosfilt(high_butterworth, y_values)

    # Return the detrended TEC vector (y-axis values).
    return detrended_tec


def butterworth(df, low_cutoff, high_cutoff, n=10, filter_order=5, plot=False, det=False, source=''):
    """
    Detrend and filter a signal using a double butterworth filter.

    :param df: Data (including a column called 'times' with the actual time of each measurement).
    :param low_cutoff: Low buttertworth filter cutoff.
    :param high_cutoff: High butterworth filter cutoff.
    :param n: Rolling mean window size.
    :param filter_order: Order of the butterworth filter.
    :param plot: Plot (True) or not (False).
    :param det: Detrend the data.
    :param source: Optional - Name of the source (e.g., 'Uranium-235').
    :return: x, y: Two vectors containing time vs. filtered/detrended energy.
    """

    # Plot original data.
    times, radiation_levels = list(df['time']), list(df['energy'])
    if plot:
        plt.plot(times, radiation_levels, linewidth=0.1)
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (kEv)')
        plt.title('Raw Data - {} Simulation'.format(source))
        plt.show()

    # Detrend the data.
    if det:
        detrended = detrend(times, radiation_levels, 1)
        filtered = detrended
    else:
        filtered = radiation_levels

    # Plot detrended data.
    if plot:
        plt.plot(times, filtered, linewidth=0.1)
        plt.xlabel('Time (s)')
        plt.ylabel('Detrended Energy (kEv)')
        plt.show()

    # Fix low == 0 or high == 0.
    if low_cutoff == 0:
        low_cutoff = None
    if high_cutoff == 0:
        high_cutoff = None

    # Filter the data.
    if low_cutoff is not None:
        filtered = detrending_high_pass(filtered, cutoff=low_cutoff, order=filter_order)
    if high_cutoff is not None:
        filtered = detrending_low_pass(filtered, cutoff=high_cutoff, order=filter_order)

    # Using abs values.
    x = times
    if det:
        filtered = detrend(x, filtered, 1, absolute=False)

    # Determine the deviation form the mean.
    y = pd.Series(filtered).rolling(n).mean()
    y = abs(y)

    # Plot.
    if plot:
        plt.plot(x, y, linewidth=0.4)
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (kEv)')
        plt.suptitle('Filtered Data - {} Simulation'.format(source))
        plt.title('Butterworth passband between 0 Hz and 1 Hz.')
        plt.show()

    return list(x)[n - 1:], list(y)[n - 1:]


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


# Save wavelet csv.
# for i in range(104900, ):
# i = 107309
# df = pd.read_csv('data\\training\\{}.csv'.format(i), header=None)
# df = add_time_column(df)
# sig = convert_to_signal(df)
# sig.to_csv('data\\wavelet_csv\\{}.csv'.format(i), header=False, index=False)
