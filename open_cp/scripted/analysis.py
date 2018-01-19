"""
analysis.py
~~~~~~~~~~~

Various routines to perform standard analysis, and/or visualisation, tasks.
"""

import matplotlib.pyplot as _plt
import matplotlib.collections as _mpl_collections
import descartes as _descartes
import csv as _csv
import collections as _collections
import scipy.stats as _stats
import open_cp.plot as _plot
import numpy as _np

def _add_outline(loaded, ax):
    p = _descartes.PolygonPatch(loaded.geometry, fc="none", ec="black")
    ax.add_patch(p)
    ax.set_aspect(1)

PredictionKey = _collections.namedtuple("PredictionKey", "name details")

def _split_by_comma_not_in_brackets(name):
    bracket_count = 0
    out = ""
    for c in name:
        if c == "(":
            bracket_count += 1
        elif c == ")":
            bracket_count -= 1
        if c == ",":
            if bracket_count == 0:
                yield out
                out = ""
            else:
                out += c
        else:
            out += c
    yield out

def parse_key_details(details):
    """Take a dictionary of "details", as returned by
    :func:`parse_prediction_key`, and splits recursively into dictionaries.
    """
    out = {}
    for k,v in details.items():
        try:
            name, dets = parse_prediction_key(v)
            out[k] = {name:dets}
        except:
            out[k] = v
    return out    

def parse_prediction_key(key):
    """The "name" or "key" of a predictor is assumed to be like:
     `ProHotspotCtsProvider(Weight=Classic(sb=400, tb=8), DistanceUnit=150)`
    
    Parse this into a :class:`PredictionKey` instance, where
    - `name` == "ProHotspotCtsProvider"
    - `details` will be the dict: {"Weight" : "Classic(sb=400, tb=8)",
                                      "DistanceUnit" : 150}
    (Attempts to parse to ints or floats if possible).
    """
    if "(" not in key:
        return PredictionKey(key, {})
    
    i = key.index("(")
    name = key[:i].strip()
    dets = key[i+1:-1]

    dets = [x.strip() for x in _split_by_comma_not_in_brackets(dets)]
    details = {}
    for x in dets:
        if "=" not in x:
            key, value = x, None
        else:
            i = x.index("=")
            key = x[:i].strip()
            value = x[i+1:].strip()
            try:
                value = int(value)
            except ValueError:
                pass
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
        details[key] = value
    
    return PredictionKey(name, details)

def plot_prediction(loaded, prediction, ax):
    """Visualise a single prediction.

    :param loaded: Instance of :class:`Loader`
    :param prediction: The prediction to plot
    :param ax: `matplotlib` Axis object to draw to.
    """
    _add_outline(loaded, ax)
    m = ax.pcolor(*prediction.mesh_data(), prediction.intensity_matrix, cmap="Greys")
    _plt.colorbar(m, ax=ax)

def _set_standard_limits(loaded, ax):
    xmin, ymin, xmax, ymax = loaded.geometry.bounds
    d = max(xmax - xmin, ymax - ymin) / 20
    ax.set(xlim=[xmin-d, xmax+d], ylim=[ymin-d, ymax+d])

def plot_data_scatter(loaded, ax):
    """Produce a scatter plot of the input data.
    
    :param loaded: Instance of :class:`Loader`
    :param ax: `matplotlib` Axis object to draw to.
    """
    _add_outline(loaded, ax)
    ax.scatter(*loaded.timed_points.coords, marker="x", linewidth=1, color="black", alpha=0.5)
    _set_standard_limits(loaded, ax)

def plot_data_grid(loaded, ax):
    """Produce a plot of masked grid we used.
    
    :param loaded: Instance of :class:`Loader`
    :param ax: `matplotlib` Axis object to draw to.
    """
    _add_outline(loaded, ax)
    pc = _mpl_collections.PatchCollection(_plot.patches_from_grid(loaded.grid),
        facecolors="none", edgecolors="black")
    ax.add_collection(pc)
    _set_standard_limits(loaded, ax)

def _open_text_file(filename, callback):
    need_close = False
    if isinstance(filename, str):
        file = open(filename, "rt", newline="")
        need_close = True
    else:
        file = filename
    try:
        return callback(file)
    finally:
        if need_close:
            file.close()


def hit_counts_to_beta(csv_file):
    """Using the data from the csv_file, return the beta distributed posterior
    given the hit count data.  This gives an indication of the "hit rate" and
    its variance.

    :param csv_file: Filename to load, or file-like object

    :return: Dictionary from prediction name to dictionary from coverage level
      to a :class:`scipy.stats.beta` instance.
    """
    def func(file):
        reader = _csv.reader(file)
        header = next(reader)
        if header[:4] != ["Predictor", "Start time", "End time" ,"Number events"]:
            raise ValueError("Input file is not from `HitCountSave`")
        coverages = [int(x[:-1]) for x in header[4:]]

        counts = _collections.defaultdict(int)
        hits = _collections.defaultdict(lambda : _collections.defaultdict(int))
        for row in reader:
            name = row[0]
            counts[name] += int(row[3])
            for cov, value in zip(coverages, row[4:]):
                hits[name][cov] += int(value)

        betas = {name : dict() for name in counts}
        for name in counts:
            for cov in coverages:
                a = hits[name][cov]
                b = counts[name] - a
                betas[name][cov] = _stats.beta(a, b)

        return betas
    
    return _open_text_file(csv_file, func)

def single_hit_counts_to_beta(hit_counts):
    """Convert a dictionary of hit_counts to beta distributed posteriors.
    
    :param hit_counts: Dictionary from arbitrary keys to dictionarys from
      coverage level to pairs `(hit_count, total_count)`.
      
    :return: Dictionary from coverage levels to :class:`scipy.stats.beta`
      instances.
    """
    total_counts = {}
    for key, cov_to_counts in hit_counts.items():
        for cov, (hit, total) in cov_to_counts.items():
            if cov not in total_counts:
                total_counts[cov] = (0, 0)
            total_counts[cov] = total_counts[cov][0] + hit, total_counts[cov][1] + total
    
    return {k : _stats.beta(a, b-a) for k, (a,b) in total_counts.items()}

def plot_betas(betas, ax, coverages=None, plot_sds=True):
    """Plot hit rate curves using the data from :func:`hit_counts_to_beta`.
    Plots the median and +/-34% (roughly a +/- 1 standard deviation) of the
    posterior estimate of the hit-rate probability.

    :param betas: Dict as from :func:`hit_counts_to_beta`.
    :param ax: `matplotlib` Axis object to draw to.
    :param coverages: If not `None`, plot only these coverages.
    :param plot_sds: If `False` then omit the "standard deviation" ranges.
    """
    if coverages is not None:
        coverages = list(coverages)
    for name, data in betas.items():
        if coverages is None:
            x = _np.sort(list(data))
        else:
            x = _np.sort(coverages)
        y = [data[xx].ppf(0.5) for xx in x]
        ax.plot(x,y,label=name)
        if plot_sds:
            y1 = [data[xx].ppf(0.5 - 0.34) for xx in x]
            y2 = [data[xx].ppf(0.5 + 0.34) for xx in x]
            ax.fill_between(x,y1,y2,alpha=0.5)
    ax.legend()
    ax.set(xlabel="Coverage (%)", ylabel="Hit rate (probability)")

def _mean_or_zero(beta_dist):
    if beta_dist.args[0] == 0:
        return 0
    return beta_dist.mean()

def compute_betas_means_against_max(betas, coverages=None):
    """Compute hit rate curves using the data from :func:`hit_counts_to_beta`.
    We use the mean "hit rate" and normalise against the maximum hit rate
    at that coverage from any prediction.
    
    :param betas: Dict as from :func:`hit_counts_to_beta`.
    :param coverages: If not `None`, plot only these coverages.
    
    :return: Pair `(x, d)` where `x` is the coverage values used, and 
      `d` is a dictionary from `betas` to list of y values.
    """
    if coverages is not None:
        x = _np.sort(list(coverages))
    else:
        data = next(iter(betas.values()))
        x = _np.sort(list(data))

    ycs = dict()
    for name, data in betas.items():
        ycs[name] = [_mean_or_zero(data[xx]) for xx in x]
    maximum = [ max(ycs[k][i] for k in ycs) for i in range(len(x)) ]

    def div_or_zero(y, m):
        if m == 0:
            return 0
        return y / m

    return x, {k : [div_or_zero(y, m) for y,m in zip(ycs[k], maximum)] for k in ycs}

def plot_betas_means_against_max(betas, ax, coverages=None):
    """Plot hit rate curves using the data from :func:`hit_counts_to_beta`.
    We use the mean "hit rate" and normalise against the maximum hit rate
    at that coverage from any prediction.
    
    :param betas: Dict as from :func:`hit_counts_to_beta`.
    :param ax: `matplotlib` Axis object to draw to.
    :param coverages: If not `None`, plot only these coverages.
    
    :return: Dictionary from keys of `betas` to list of y values.
    """
    x, normed = compute_betas_means_against_max(betas, coverages)
    for name, y in normed.items():
        ax.plot(x,y,label=name)
    ax.set(ylabel="Fraction of maximum hit rate", xlabel="Coverage (%)")
    return normed
