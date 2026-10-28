import warnings
from typing import List

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

# one or more simple plots of y data vs x data on shared axes
from permutect.data.count_binning import LOGIT_BIN_SKIP

# Filter the specific warning
warnings.filterwarnings(
    "ignore",
    message="No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.",
)


def legend_if_applicable(ax):
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()


def simple_plot(x_y_lab_tuples, x_label, y_label, title):
    fig = plt.figure()
    curve = fig.gca()
    for x, y, lab in x_y_lab_tuples:
        if lab is not None:
            curve.plot(x, y, label=lab)
        else:
            curve.plot(x, y)
    curve.set_title(title)
    curve.set_xlabel(x_label)
    curve.set_ylabel(y_label)
    legend_if_applicable(curve)
    return fig, curve


def simple_plot_on_axis(ax, x_y_lab_tuples, x_label, y_label):
    for x, y, lab in x_y_lab_tuples:
        if lab is not None:
            ax.plot(x, y, label=lab)
        else:
            ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    legend_if_applicable(ax)


# x bounds has length 1 greater than values's 0th dimension
# y bounds has length 1 greater than values's 1st dimension
def color_plot_2d_on_axis(ax, x_bounds, y_bounds, values, x_label, y_label, vmin: float = None, vmax: float = None):
    mesh = ax.pcolormesh(x_bounds, y_bounds, values, vmin=vmin, vmax=vmax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return mesh


def simple_histograms_on_axis(ax, list_of_histogram_data, list_of_labels, num_bins):
    ax.hist(list_of_histogram_data, bins=num_bins, alpha=0.5, label=list_of_labels)


# apply grouped bar plot to an axis (subplot) object
# heights by category is a dict of category to bar heights, where the nth bar height
# corresponds to the nth x label
def grouped_bar_plot_on_axis(ax, heights_by_category, x_labels, y_label):
    spacing = 5
    bar_width = 0.7 * spacing / len(heights_by_category)

    for n, (category, heights) in enumerate(heights_by_category.items()):
        offset = n * bar_width
        x_positions = [offset + spacing * i for i in range(len(heights))]
        ax.bar(x_positions, heights, width=bar_width, edgecolor="white", label=category)

    # Add xticks on the middle of the group bars
    # plt.xlabel('group', fontweight='bold')
    ticks_offset = bar_width * len(heights_by_category) / 2
    ax.set_xticks([ticks_offset + spacing * i for i in range(len(x_labels))], labels=x_labels)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel(y_label)
    legend_if_applicable(ax)


# heights by category is a dict of category to bar heights, where the nth bar height
# corresponds to the nth x label
def grouped_bar_plot(heights_by_category, x_labels, y_label):
    fig, ax = plt.subplots()
    grouped_bar_plot_on_axis(ax, heights_by_category, x_labels, y_label)
    return fig, ax


def plot_roc_on_axis(list_of_tuple_lists, curve_labels, axis, sens_prec: bool, thresholds: List[float]):
    # input is a list of lists
    # each constituent list is of tuples (threshold, nonartifact metric, artifact metric)
    x_y_lab_tuples = []
    small_dots, big_dots = [], []
    for given_threshold, thresh_and_accs, curve_label in zip(thresholds, list_of_tuple_lists, curve_labels):
        nonart_metrics = [x[1] for x in thresh_and_accs]
        art_metrics = [x[2] for x in thresh_and_accs]
        x_y_lab_tuples.append((nonart_metrics, art_metrics, curve_label))

        for threshold, nonart_metric, art_metric in thresh_and_accs:
            if abs(threshold) < 0.001:
                big_dots.append((nonart_metric, art_metric, "rs"))  # red square
            elif given_threshold is not None and abs(threshold - given_threshold) <= LOGIT_BIN_SKIP / 2:
                big_dots.append((nonart_metric, art_metric, "kd"))  # black diamond
            else:
                small_dots.append((nonart_metric, art_metric, "go"))  # green circle

    simple_plot_on_axis(
        axis,
        x_y_lab_tuples,
        "sensitivity" if sens_prec else "non-artifact accuracy",
        "precision" if sens_prec else "artifact accuracy",
    )
    for x, y, spec in small_dots:
        axis.plot(x, y, spec, markersize=2, label="")  # point
    for x, y, spec in big_dots:
        axis.plot(x, y, spec, markersize=6, label="")  # point


def tidy_subplots(
    figure: Figure,
    axes,
    x_label: str = None,
    y_label: str = None,
    col_labels: List[str] = None,
    row_labels: List[str] = None,
    keep_axes_tick_labels=False,
):
    """
    Combines various tidying operations on figures with subplots
    1.  Removes the individual axis legends and replaces with a single figure legend.  This assumes
        that all axes have the same lines.
    2.  Show x (y) labels and tick labels only in bottom row (leftmost column)
    3.  Apply column headings and row labels
    4.  Apply overall x and y labels to the figure as a whole

    figure matplotlib.figure.Figure
    axes:   2D array of matplotlib.axes.Axes

    We assume these have been generated together via figure, axes = plt.subplots(. . .)

    """
    handles, labels = figure.get_axes()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center")

    for ax in figure.get_axes():
        if not keep_axes_tick_labels:
            ax.label_outer()  # y tick labels only shown in leftmost column, x tick labels only shown on bottom row
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend().set_visible(False)  # hide the redundant identical subplot legends

        # remove the subplot labels and title -- these will be given manually to the whole figure and to the outer rows
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(None)

    if x_label is not None:
        figure.supxlabel(x_label)

    if y_label is not None:
        figure.supylabel(y_label)

    if row_labels is not None:
        assert len(row_labels) == len(axes)
        for row_idx, label in enumerate(row_labels):
            axes[row_idx][0].set_ylabel(label)  # note that we use row 0 and set_title to make this a column heading

    if col_labels is not None:
        assert len(col_labels) == len(axes[0])
        for col_idx, label in enumerate(col_labels):
            axes[0][col_idx].set_title(label)

    figure.tight_layout()
