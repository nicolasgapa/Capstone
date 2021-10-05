# Imports.
from signal_processing import add_time_column, butterworth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Butterworth plotting.
def generate_results(output_file, butterworth_order=5, low_threshold=0, high_threshold=1):
    labels_file = 'data\\trainingAnswers.csv'
    labels = pd.read_csv(labels_file)
    sources = {1: 'Uranium', 2: 'Plutonium', 3: 'Iodine', 4: 'Cobalt', 5: 'Technetium', 6: 'HEU + Tech'}
    results = []
    ct = 0
    for i in range(104901, 109701):
        label, peak = labels[labels['RunID'] == i].to_numpy()[0][1:]
        df = pd.read_csv('data\\training\\{}.csv'.format(i), header=None)
        df = add_time_column(df)
        df.columns = [0, 'energy', 'time']
        x, y = butterworth(df, low_threshold, high_threshold, n=1000, plot=False, det=True, source=sources[label],
                           filter_order=butterworth_order)
        local_df = pd.DataFrame(np.array([x, y]).T)
        local_df = local_df[local_df[0] >= 30]
        detection = local_df.loc[local_df[1].idxmax()][0]
        error = (abs(detection - peak)) / (max(x) - 30)
        results.append([label, peak, detection, peak - detection, max(x), error])
        print('{}: Source: {}. Peak at {}s. Detection at {}. Difference: {}. Error: {}%'.format(ct + 1, label, peak,
                                                                                                round(detection, 1),
                                                                                                round(
                                                                                                    abs(peak - detection),
                                                                                                    1),
                                                                                                round(error * 100, 1)))
        # if abs(detection - peak) > 5:
        #    x, y = butterworth(df, 0, 1, n=1000, plot=True, det=True, source=sources[label])
        if ct % 100 == 0:
            df1 = pd.DataFrame(results)
            df1.to_csv(output_file)
        ct += 1
    df1 = pd.DataFrame(results)
    df1.to_csv(output_file)


def test_results(results_file):
    # Open file.
    df = pd.read_csv(results_file)
    df = df.iloc[:, [1, 4]]
    df.columns = ['label', 'error']
    df['error'] = abs(df['error'])

    # Remove outliers.
    summary = df.groupby('label').mean()
    q25s = df.groupby('label').quantile(0.25)
    q75s = df.groupby('label').quantile(0.75)
    summary['q25'] = q25s['error']
    summary['q75'] = q75s['error']
    summary['iqr'] = abs(summary['q25'] - summary['q75'])
    summary['lower_limit'] = -10  # summary['q25'] - summary['iqr'] * 1.5
    summary['upper_limit'] = 10 # summary['q75'] + summary['iqr'] * 1.5

    # Filter per source.
    u, p, i, c, t, h = df[df['label'] == 1], df[df['label'] == 2], df[df['label'] == 3], df[df['label'] == 4], df[
        df['label'] == 5], df[df['label'] == 6]
    means = [abs(e['error']).mean() for e in [u, p, i, c, t, h]]
    print('Before:', means)

    u = u[u['error'] >= summary['lower_limit'][1]]
    u = u[u['error'] <= summary['upper_limit'][1]]
    p = p[p['error'] >= summary['lower_limit'][2]]
    p = p[p['error'] <= summary['upper_limit'][2]]
    i = i[i['error'] >= summary['lower_limit'][3]]
    i = i[i['error'] <= summary['upper_limit'][3]]
    c = c[c['error'] >= summary['lower_limit'][4]]
    c = c[c['error'] <= summary['upper_limit'][4]]
    t = t[t['error'] >= summary['lower_limit'][5]]
    t = t[t['error'] <= summary['upper_limit'][5]]
    h = h[h['error'] >= summary['lower_limit'][6]]
    h = h[h['error'] <= summary['upper_limit'][6]]

    # Print means.
    means = [abs(e['error']).mean() for e in [u, p, i, c, t, h]]
    print('After:', means)
    print('Detections', [len(e) * 100 / 800 for e in [u, p, i, c, t, h]])


generate_results('results_order_1_passband_0_05.csv', butterworth_order=1, low_threshold=0, high_threshold=0.5)
test_results('results_order_1_passband_0_05.csv')
