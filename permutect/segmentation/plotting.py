"""
A plot of a sample's segmentation by minor allele fraction (MAF), like GATK's PlotModeledSegments: the alt fraction of
each site's reads against its genomic position.  Across each segment, a line marks its MAF inside a shaded box of its
interval, which has most of the MAF's posterior, and the same is mirrored at 1 - MAF.  Contigs are laid end to end in
their order in the data, each as wide as its last site's position, with a shaded band behind every other one.  The
color of the sites alternates from one segment to the next.
"""

import math

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from permutect.segmentation.emissions import Pileups
from permutect.segmentation.segmenter import Segment

# the first two slots of a categorical palette that stays distinguishable with color vision deficiencies
SITE_COLORS = ("#2a78d6", "#eb6834")
BAR_COLOR = "#0b0b0b"
# the boxes of the MAFs' posterior intervals are shaded, not outlined, so that short segments don't become black lines
BOX_ALPHA = 0.25
SURFACE_COLOR = "#fcfcfb"
BAND_COLOR = "#f0efec"
AXIS_COLOR = "#c3c2b7"
LABEL_COLOR = "#52514e"
# a whole genome can have tens of millions of sites, so plot a random subset of at most this many
MAX_PLOTTED_SITES = 1_000_000


def plot_segmentation(
    path,
    sample: str,
    contigs_c: list[str],
    lengths_c: np.ndarray,
    positions_n: np.ndarray,
    pileups: Pileups,
    segments: list[Segment],
) -> Figure:
    """
    Plot the sites and segments of a sample to a file whose format matplotlib infers from its extension, eg .png.  The
    sites of each contig are consecutive, with lengths_c[c] sites in contig c, in order of position.
    """
    starts_c = np.cumsum(lengths_c) - lengths_c
    widths_c = np.array([positions_n[start + length - 1] for start, length in zip(starts_c, lengths_c)])
    offsets_c = np.cumsum(widths_c) - widths_c
    offsets = dict(zip(contigs_c, offsets_c))
    contig_slices = [slice(start, start + length) for start, length in zip(starts_c, lengths_c)]
    sites = {contig: positions_n[contig_slice] for contig, contig_slice in zip(contigs_c, contig_slices)}

    # segments of sites, as opposed to the gap segments between them, which have no posterior
    bars = [s for s in segments if not math.isnan(s.maf_low)]

    # sites alternate colors from one segment to the next, counting segments over all contigs
    segment_parities_n = np.zeros(len(positions_n), dtype=np.int64)
    num_previous_bars = 0
    for contig, start, length in zip(contigs_c, starts_c, lengths_c):
        bar_starts = np.array([bar.start for bar in bars if bar.contig == contig])
        bar_indices = np.searchsorted(bar_starts, sites[contig], side="right") - 1
        segment_parities_n[start : start + length] = (num_previous_bars + bar_indices) % 2
        num_previous_bars += len(bar_starts)

    x_n = positions_n + np.repeat(offsets_c, lengths_c)
    alt_fractions_n = (pileups.alt_counts_n / pileups.depths_n).numpy()
    plotted_n = np.arange(len(positions_n))
    if len(plotted_n) > MAX_PLOTTED_SITES:
        plotted_n = np.sort(np.random.default_rng(0).choice(plotted_n, MAX_PLOTTED_SITES, replace=False))

    fig = Figure(figsize=(16, 5), facecolor=SURFACE_COLOR)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE_COLOR)
    for c in range(1, len(contigs_c), 2):
        ax.axvspan(offsets_c[c], offsets_c[c] + widths_c[c], color=BAND_COLOR, linewidth=0, zorder=0)

    colors = np.array(SITE_COLORS)[segment_parities_n[plotted_n]]
    ax.scatter(x_n[plotted_n], alt_fractions_n[plotted_n], s=2, c=colors, alpha=0.4, linewidths=0, rasterized=True)

    if bars:
        bar_starts = np.array([offsets[bar.contig] + bar.start for bar in bars])
        bar_ends = np.array([offsets[bar.contig] + bar.end for bar in bars])
        mafs = np.array([bar.maf for bar in bars])
        bar_ys = np.concatenate((mafs, 1 - mafs))
        ax.hlines(bar_ys, np.tile(bar_starts, 2), np.tile(bar_ends, 2), colors=BAR_COLOR, linewidth=2)
        boxes = [
            Rectangle((offsets[bar.contig] + bar.start, low), bar.end - bar.start, high - low)
            for bar in bars
            for low, high in ((bar.maf_low, bar.maf_high), (1 - bar.maf_high, 1 - bar.maf_low))
        ]
        ax.add_collection(PatchCollection(boxes, facecolor=BAR_COLOR, alpha=BOX_ALPHA, linewidth=0))

    # contig names under their middles, staggered over two rows so that short neighboring contigs don't collide
    ax.set_xticks(offsets_c + widths_c / 2)
    ax.set_xticklabels([contig if c % 2 == 0 else "\n" + contig for c, contig in enumerate(contigs_c)], fontsize=8)
    ax.tick_params(axis="x", length=0, colors=LABEL_COLOR)
    ax.tick_params(axis="y", colors=LABEL_COLOR)
    ax.set_xlim(0, max(np.sum(widths_c), 1))
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("contig", color=LABEL_COLOR)
    ax.set_ylabel("alt allele fraction", color=LABEL_COLOR)
    ax.set_title(sample, loc="left", fontsize=12, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS_COLOR)

    site_handle = Line2D([], [], linestyle="", marker="o", markersize=4, color=SITE_COLORS[0])
    bar_handle = Line2D([], [], color=BAR_COLOR, linewidth=2)
    box_handle = Rectangle((0, 0), 1, 1, facecolor=BAR_COLOR, alpha=BOX_ALPHA, linewidth=0)
    handles = [site_handle, bar_handle, box_handle]
    labels = ["site, colored by segment", "segment MAF and 1 - MAF", "80% posterior interval"]
    ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1, 1), ncol=3, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
