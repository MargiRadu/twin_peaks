import numpy as np
import sys
import matplotlib.pyplot as plt

from datetime import date
import yfinance as yf

from analysis import get_landmarks_2008, get_landmarks_2020, find_peaks_troughs, smart_filter, estimate_trough
from plots import plot_simple, plot_landmarks_2008, plot_landmarks_2020, plot_points, plot_landmarks_arbitrary


def analyze_2008(ax, ticker):
    df = yf.download(ticker,
                     start='2007-06-01',
                     end='2009-12-31',
                     interval='1d',
                     progress=False)

    close_prices = df['Close']
    # close_prices = (close_prices - close_prices.mean()) / close_prices.std()

    plot_simple(ax, close_prices, color='tab:blue')

    close_prices_filtered = smart_filter(close_prices, sigma=20.0)
    plot_simple(ax, close_prices_filtered, color='tab:orange')

    # Find the landmarks
    major_peak_ix, minor_trough_ix, major_trough_ix = get_landmarks_2008(close_prices_filtered, close_prices,
                                                                         local_window_size=41,
                                                                         peak_window_size=11,
                                                                         min_trough_distance=1)
    plot_landmarks_2008(ax, close_prices, major_peak_ix, minor_trough_ix, major_trough_ix)

    peak_ixs, troughs_ixs = find_peaks_troughs(close_prices_filtered, window_size=5)
    plot_points(ax, close_prices_filtered, peak_ixs, color='r')
    plot_points(ax, close_prices_filtered, troughs_ixs, color='b')

    return close_prices, major_peak_ix, minor_trough_ix, major_trough_ix


def analyze_2020(ax, ticker):
    today = date.today().strftime("%Y-%m-%d")
    df = yf.download(ticker,
                     start='2019-11-01',
                     end=today,
                     interval='1d',
                     progress=False)

    close_prices = df['Close']
    # close_prices = (close_prices - close_prices.mean()) / close_prices.std()

    plot_simple(ax, close_prices, color='tab:blue')

    close_prices_filtered = smart_filter(close_prices, sigma=20.0)
    plot_simple(ax, close_prices_filtered, color='tab:orange')

    # Find the landmarks
    major_peak_ix, minor_trough_ix = get_landmarks_2020(close_prices_filtered, close_prices,
                                                        local_window_size=41,
                                                        peak_window_size=11,
                                                        min_trough_distance=1)
    plot_landmarks_2020(ax, close_prices, major_peak_ix, minor_trough_ix)

    peak_ixs, troughs_ixs = find_peaks_troughs(close_prices_filtered, window_size=5)
    plot_points(ax, close_prices_filtered, peak_ixs, color='r')
    plot_points(ax, close_prices_filtered, troughs_ixs, color='b')

    return close_prices, major_peak_ix, minor_trough_ix


class TextDrawer:

    def __init__(self, ax):
        self.ax = ax
        self.current_ix = 0
        self.margin = 0.1
        self.row_width = 0.05
        self.font_size = 13

    def text(self, text_str):
        col_pos, row_pos = self.margin, 1 - (self.margin + self.current_ix * self.row_width)
        self.ax.text(col_pos, row_pos, text_str, fontsize=13)

        self.current_ix += 1

    def newline(self, n=1):
        for _ in range(n):
            self.text('')


def draw_info(text_ax, prices_2008, major_peak_2008, minor_trough_2008, major_trough_2008,
              prices_2020, major_peak_2020, minor_trough_2020, major_trough_2020_naive, major_trough_2020_normed):
    def future_time(time_index, ix):
        if ix > len(time_index):
            last_time = time_index[-1]
            last_ix = len(time_index) - 1
            return last_time + np.timedelta64(int((ix - last_ix) * 1.4), 'D')
        else:
            return time_index[ix]

    def index_future_date_str(time_index, ix):
        date = future_time(time_index, ix)
        return date.strftime("%d %b %Y")

    def index_time_delta_str(time_index, t1, t2):
        weeks = (future_time(time_index, t2) - time_index[t1]).days / 7
        return f'{weeks:.1f}'

    def index_date_str(time_index, ix):
        return time_index[ix].strftime("%d %b %Y")

    naive_estim_str = index_future_date_str(prices_2020.index, major_trough_2020_naive)
    norm_estim_str = index_future_date_str(prices_2020.index, major_trough_2020_normed)

    peak_2020 = index_date_str(prices_2020.index, major_peak_2020)
    first_trough_2020 = index_date_str(prices_2020.index, minor_trough_2020)

    peak_2008 = index_date_str(prices_2008.index, major_peak_2008)
    first_trough_2008 = index_date_str(prices_2008.index, minor_trough_2008)
    second_trough_2008 = index_date_str(prices_2008.index, major_trough_2008)

    first_peak_bottom_2008_str = index_time_delta_str(prices_2008.index, major_peak_2008, minor_trough_2008)
    second_peak_bottom_2008_str = index_time_delta_str(prices_2008.index, major_peak_2008, major_trough_2008)

    first_peak_bottom_2020_str = index_time_delta_str(prices_2020.index, major_peak_2020, minor_trough_2020)
    second_peak_bottom_2020_str_naive = index_time_delta_str(prices_2020.index, major_peak_2020,
                                                             major_trough_2020_naive)
    second_peak_bottom_2020_str_norm = index_time_delta_str(prices_2020.index, major_peak_2020,
                                                            major_trough_2020_normed)

    first_drop_2008 = int(100 * prices_2008.iloc[minor_trough_2008] / prices_2008[major_peak_2008])
    first_drop_2020 = int(100 * prices_2020.iloc[minor_trough_2020] / prices_2020[major_peak_2020])
    second_drop_2008 = int(100 * prices_2008.iloc[major_trough_2008] / prices_2008[major_peak_2008])

    price_estim_str = int(second_drop_2008 / 100 * prices_2020.iloc[major_peak_2020])

    td = TextDrawer(text_ax)
    td.text('2008')
    td.text(f'Start peak: {peak_2008}')
    td.text(f'First trough: {first_trough_2008} ({first_drop_2008}%)')
    td.text(f'Second trough: {second_trough_2008} ({second_drop_2008}%)')
    td.newline()
    td.text(f'First peak to bottom distance: {first_peak_bottom_2008_str} weeks')
    td.text(f'Second peak to bottom distance: {second_peak_bottom_2008_str} weeks')
    td.newline(2)

    td.text('2020')
    td.text(f'Start peak: {peak_2020}')
    td.text(f'First trough: {first_trough_2020} ({first_drop_2020}%)')
    td.text(f'Second trough (normalized): {norm_estim_str}')
    td.text(f'Second trough (naive): {naive_estim_str}')
    td.text(f'Estimated through price: {price_estim_str}$')

    td.newline()
    td.text(f'First peak to bottom distance: {first_peak_bottom_2020_str} weeks')
    td.text(f'Second peak to bottom distance (normalized): {second_peak_bottom_2020_str_norm} weeks')
    td.text(f'Second peak to bottom distance (naive): {second_peak_bottom_2020_str_naive} weeks')

    text_ax.axis('off')


def run_for_ticker(ticker, ax_2008, ax_2020, text_ax):
    prices_2008, major_peak_2008, minor_trough_2008, major_trough_2008 = analyze_2008(ax_2008, ticker)
    prices_2020, major_peak_2020, minor_trough_2020 = analyze_2020(ax_2020, ticker)

    # Compute necessary predictions
    naive, normed = estimate_trough(prices_2008, major_peak_2008, minor_trough_2008, major_trough_2008,
                                    prices_2020, major_peak_2020, minor_trough_2020)

    plot_landmarks_arbitrary(ax_2020, prices_2020, [naive, normed], markers=['s', '<'])

    ax_2008.set_title(f'{ticker} 2008')
    ax_2020.set_title(f'{ticker} 2020')

    ax_2008.tick_params(axis='both', which='major', labelsize=7)
    ax_2020.tick_params(axis='both', which='major', labelsize=7)

    draw_info(text_ax, prices_2008, major_peak_2008, minor_trough_2008, major_trough_2008,
              prices_2020, major_peak_2020, minor_trough_2020, naive, normed)


class NavigableStockPrices:

    def __init__(self, path_to_tickers=None):
        if path_to_tickers is None:
            self._tickers = ['AAPL', 'GOOG']
        else:
            with open(path_to_tickers, 'r') as fp:
                self._tickers = fp.read().splitlines()

        self._ticker_ix = 0
        self._n_tickers = len(self._tickers)

        self.fig, (self.ax_2008, self.ax_2020, self.text_ax) = plt.subplots(1, 3, squeeze=True)
        self.fig.canvas.mpl_connect('key_press_event', self.handler)

        self.update_for_ticker(self._tickers[self._ticker_ix])

    def previous_ticker(self):
        self._ticker_ix -= 1
        if self._ticker_ix < 0:
            self._ticker_ix = self._n_tickers - 1

        return self._tickers[self._ticker_ix]

    def next_ticker(self):
        self._ticker_ix += 1
        if self._ticker_ix >= self._n_tickers:
            self._ticker_ix = 0

        return self._tickers[self._ticker_ix]

    def handler(self, event):
        if event.key == 'd' or event.key == 'right':
            ticker = self.next_ticker()
            while True:
                try:
                    self.update_for_ticker(ticker)
                    break
                except:
                    print(f'Something went wrong when processing {ticker}. Skipping...')
                    ticker = self.next_ticker()

        elif event.key == 'a' or event.key == 'left':
            ticker = self.previous_ticker()
            while True:
                try:
                    self.update_for_ticker(ticker)
                    break
                except:
                    print(f'Something went wrong when processing {ticker}. Skipping...')
                    ticker = self.previous_ticker()

        elif event.key == 'escape':
            print('Bye!')
            sys.exit(0)

    def update_for_ticker(self, ticker):
        self.ax_2008.clear()
        self.ax_2020.clear()
        self.text_ax.clear()

        run_for_ticker(ticker, self.ax_2008, self.ax_2020, self.text_ax)

        self.fig.canvas.draw()


if __name__ == '__main__':
    interface = NavigableStockPrices(path_to_tickers='tickers.txt')
    plt.show()
