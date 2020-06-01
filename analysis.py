import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter


def _find_closest_peak(series, ix, window_size, ix_finder_func):
    half_window_size = (window_size - 1) // 2
    window = np.arange(max(0, ix - half_window_size), min(ix + half_window_size, len(series) + 1))
    ix_in_window = ix_finder_func(series[window])
    return window[ix_in_window]


def find_peaks_troughs(series, window_size=5):
    half_window_size = (window_size - 1) // 2
    values = series.values

    peaks = []
    trough = []

    for value_ix, value in enumerate(values):
        window = values[max(0, value_ix - half_window_size): value_ix + half_window_size + 1]
        if value == np.min(window):
            trough.append(value_ix)
        if value == np.max(window):
            peaks.append(value_ix)

    return np.array(peaks), np.array(trough)


def get_landmarks_2008(series_filtered, series_original, local_window_size, peak_window_size=5, min_trough_distance=15):
    values = series_filtered.values
    peak_ixs, troughs_ixs = find_peaks_troughs(series_filtered, peak_window_size)

    peaks = values[peak_ixs]
    troughs = values[troughs_ixs]

    # Find the major trough
    major_trough_ix = troughs_ixs[np.argmin(troughs)]

    # Find the minor trough
    minor_trough_ix = troughs_ixs[troughs_ixs < major_trough_ix - min_trough_distance][-1]

    # Find the major peak
    major_peak_ix = peak_ixs[peak_ixs < minor_trough_ix][-1]

    # Find the corresponding peaks and troughs in the original series
    original_major_trough_ix = _find_closest_peak(series_original.values, major_trough_ix,
                                                  window_size=local_window_size,
                                                  ix_finder_func=np.argmin)
    original_minor_trough_ix = _find_closest_peak(series_original.values, minor_trough_ix,
                                                  window_size=local_window_size,
                                                  ix_finder_func=np.argmin)
    original_major_peak_ix = _find_closest_peak(series_original.values, major_peak_ix,
                                                window_size=local_window_size,
                                                ix_finder_func=np.argmax)

    return original_major_peak_ix, original_minor_trough_ix, original_major_trough_ix


def get_landmarks_2020(series_filtered, series_original, local_window_size, peak_window_size=5, min_trough_distance=15):
    values = series_filtered.values
    peak_ixs, troughs_ixs = find_peaks_troughs(series_filtered, peak_window_size)

    # Find most recent trough
    most_recent_trough_ix = troughs_ixs[-1]
    most_recent_trough_tuned_ix = _find_closest_peak(series_original.values, most_recent_trough_ix,
                                                     window_size=local_window_size,
                                                     ix_finder_func=np.argmin)

    # Find the first peak before the MRT
    major_peak_ix = peak_ixs[peak_ixs < most_recent_trough_ix][-1]
    major_peak_tuned_ix = _find_closest_peak(series_original.values, major_peak_ix,
                                             window_size=local_window_size,
                                             ix_finder_func=np.argmax)

    return major_peak_tuned_ix, most_recent_trough_tuned_ix


def smart_filter(series, sigma=15):
    values = series.values
    new_values = gaussian_filter(values, sigma=sigma)
    return pd.Series(data=np.array(new_values), index=series.index)


def estimate_trough(prices_2008, major_peak_2008, minor_trough_2008, major_trough_2008,
                    prices_2020, major_peak_2020, minor_trough_2020):
    # Estimate 1 - naive number of days
    major_trough_2020_naive = major_peak_2020 + (major_trough_2008 - major_peak_2008)

    # Estimate 2 - number of days normalized by delta_1
    delta_2008 = minor_trough_2008 - major_peak_2008
    normalized_distance_2008 = (major_trough_2008 - major_peak_2008) / delta_2008

    delta_2020 = minor_trough_2020 - major_peak_2020
    major_trough_2020_norm = major_peak_2020 + int(delta_2020 * normalized_distance_2008)

    return major_trough_2020_naive, major_trough_2020_norm
