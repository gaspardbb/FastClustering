from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def add_linear_regression(ax=None, custom_label=None, skip_first_quantile=0, skip_last_quantile=1):
    """
    Parameters
    ----------
    ax:
        Matplotlib Axes on which to add the legend
    custom_label:
        Seaborn has a really messy way of handling line labels. Pass a custom list of labels here.
    """
    if ax is None:
        ax = plt.gca()
    ax: plt.Axes

    for i, line in enumerate(ax.get_lines()):
        line: Line2D

        x, y = line.get_data()
        sorted_x = np.argsort(x)
        x, y = x[sorted_x], y[sorted_x]
        if skip_first_quantile != 0:
            q = skip_first_quantile
            x = x[int(x.shape[0] * q):]
            y = y[int(y.shape[0] * q):]
        if skip_last_quantile != 1:
            q = skip_last_quantile
            x = x[:int(x.shape[0] * q)]
            y = y[:int(y.shape[0] * q)]

        if x.size == 0 or y.size == 0:
            # This line is empty, probably seaborn great handling
            continue

        color = line.get_color()
        if custom_label is None:
            label = line.get_label()
        else:
            label = custom_label[i]

        if ax.get_xscale() == 'log':
            x = np.log10(x)
        if ax.get_yscale() == 'log':
            y = np.log10(y)
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        coef, intercept = model.coef_[0], model.intercept_
        linear_regression = lambda x: x * coef + intercept
        x_start, x_end = x[0], x[-1]
        y_start, y_end = linear_regression(x_start), linear_regression(x_end)

        if ax.get_xscale() == 'log':
            x_start, x_end = 10 ** x_start, 10 ** x_end
        if ax.get_yscale() == 'log':
            y_start, y_end = 10 ** y_start, 10 ** y_end
        ax.plot([x_start, x_end], [y_start, y_end], color=color, linestyle='-.', alpha=.2,
                label=f'reg_{label}: {coef:.2f}x + {intercept:.2f}')
    return