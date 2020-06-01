import numpy as np


def plot_simple(ax, series, color):
    ax.plot(series.index, series, color=color)


def plot_landmarks_arbitrary(ax, series, ixs, markers):
    landmark_xs = []
    landmark_ys = []

    last_time = series.index.values[-1]
    last_ix = len(series) - 1
    for ix in ixs:
        if ix > last_ix:
            x = last_time + np.timedelta64(int((ix - last_ix) * 1.4), 'D')
            y = np.min(series)
        else:
            x = series.index[ix]
            y = series.iloc[ix]
        landmark_xs.append(x)
        landmark_ys.append(y)

    for x, y, marker in zip(landmark_xs, landmark_ys, markers):
        ax.scatter(x, y, marker=marker, color='k', s=225, alpha=0.3)


def plot(ax, series, major_peak, minor_trough, major_trough):
    ixs = [major_peak, minor_trough]
    if major_trough is not None:
        ixs.append(major_trough)

    landmarks_xs = [series.index[ix] for ix in ixs]
    landmark_ys = [series.iloc[ix] for ix in ixs]
    markers = ['^', '>', '<']

    for x, y, marker in zip(landmarks_xs, landmark_ys, markers):
        ax.scatter(x, y, marker=marker, color='k', s=225)


def plot_landmarks_2008(ax, series, major_peak, minor_trough, major_trough):
    ixs = [major_peak, minor_trough, major_trough]
    landmarks_xs = [series.index[ix] for ix in ixs]
    landmark_ys = [series.iloc[ix] for ix in ixs]
    markers = ['^', '>', '<']

    for x, y, marker in zip(landmarks_xs, landmark_ys, markers):
        ax.scatter(x, y, marker=marker, color='k', s=225)


def plot_landmarks_2020(ax, series, major_peak, mr_trough):
    ixs = [major_peak, mr_trough]
    landmarks_xs = [series.index[ix] for ix in ixs]
    landmark_ys = [series.iloc[ix] for ix in ixs]
    markers = ['^', '>']

    for x, y, marker in zip(landmarks_xs, landmark_ys, markers):
        ax.scatter(x, y, marker=marker, color='k', s=225)


def plot_points(ax, series, ixs, color='k'):
    xs = [series.index[ix] for ix in ixs]
    ys = [series.iloc[ix] for ix in ixs]
    ax.scatter(xs, ys, marker='o', color=color)
