"""
seppexp
~~~~~~~

Implements the ETAS (Epidemic Type Aftershock-Sequences) model intensity
estimation scheme outlined in Mohler et al. (2015).

References
~~~~~~~~~~
Mohler et al, "Randomized Controlled Field Trials of Predictive Policing",
   Journal of the American Statistical Association (2015)
   DOI:10.1080/01621459.2015.1077710

Lewis, Mohler, "A Nonparametric EM Algorithm for Multiscale Hawkes Processes"
   in Proceedings of the 2011 Joint Statistical Meetings, pp. 1–16
   http://math.scu.edu/~gmohler/EM_paper.pdf
"""

#from . import predictors
#from . import kernels
import numpy as _np
import itertools as _itertools

def _normalise_matrix(p):
    column_sums = _np.sum(p, axis=0)
    return p / column_sums[None,:]

def p_matrix(points, omega, theta, mu):
    """Computes the probability matrix.

    :param points: A one-dimensional of the times of events, in increasing
    order.
    :param omega: The scale of the "triggering" exponential distribution
    :param theta: The rate of the "triggering" intensity
    :param mu: The background Poisson process rate.

    :return: The normalised probability matrix.
    """
    number_data_points = len(points)
    p = _np.zeros((number_data_points, number_data_points))
    for j in range(1, number_data_points):
        d = points[j] - points[0:j]
        p[0:j, j] = theta * omega * _np.exp(-omega * d)
    for j in range(number_data_points):
        p[j, j] = mu
    return _normalise_matrix(p)

def maximisation(cells, omega, theta, mu, time_duration):
    """Perform an iteration of the EM algorithm.

    :param cells: An array (of any shape) each entry of which is an array of
    times of events, in increasing order.
    :param mu: An array, of the same shape as `cells`, giving the background
    rate in each cell.
    :param time_duration: The total time range of the data.

    :return: Triple (omega, theta, mu) of new estimates.
    """
    cells, mu = _np.asarray(cells), _np.asarray(mu)
    upper_trianglar_sums = _np.zeros_like(mu)
    weighted_upper_trianglar_sums = _np.zeros_like(mu)
    diagonal_sums = _np.zeros_like(mu)
    event_counts = _np.zeros_like(mu)
    
    for index in _itertools.product(*[list(range(i)) for i in cells.shape]):
        times = cells[index]
        p = p_matrix(times, omega, theta, mu[index])
        diag_sum = _np.sum(_np.diag(p))
        diagonal_sums[index] = diag_sum
        upper_trianglar_sums[index] = _np.sum(p) - diag_sum
        weighted_p = p * (times[None, :] - times[:, None])
        weighted_upper_trianglar_sums[index] = _np.sum(weighted_p)
        event_counts[index] = len(times)
    
    omega = _np.sum(upper_trianglar_sums) / _np.sum(weighted_upper_trianglar_sums)
    theta = _np.sum(upper_trianglar_sums) / _np.sum(event_counts)
    mu = diagonal_sums / time_duration

    return (omega, theta, mu)

def maximisation_corrected(cells, omega, theta, mu, time_duration):
    """TODO Perform an iteration of the EM algorithm.

    :param cells: An array (of any shape) each entry of which is an array of
    times of events, in increasing order.
    :param mu: An array, of the same shape as `cells`, giving the background
    rate in each cell.
    :param time_duration: The total time range of the data.

    :return: Triple (omega, theta, mu) of new estimates.
    """
    cells, mu = _np.asarray(cells), _np.asarray(mu)
    upper_trianglar_sums = _np.zeros_like(mu)
    weighted_upper_trianglar_sums = _np.zeros_like(mu)
    diagonal_sums = _np.zeros_like(mu)
    event_counts = _np.zeros_like(mu)
    
    for index in _itertools.product(*[list(range(i)) for i in cells.shape]):
        times = cells[index]
        p = p_matrix(times, omega, theta, mu[index])
        diag_sum = _np.sum(_np.diag(p))
        diagonal_sums[index] = diag_sum
        upper_trianglar_sums[index] = _np.sum(p) - diag_sum
        weighted_p = p * (times[None, :] - times[:, None])
        dt = time_duration - times
        dtt = _np.exp(-omega * dt)
        weighted_upper_trianglar_sums[index] = (_np.sum(weighted_p) + 
            theta * _np.sum(dt * dtt))
        event_counts[index] = len(times) - _np.sum(dtt)
    
    omega = _np.sum(upper_trianglar_sums) / _np.sum(weighted_upper_trianglar_sums)
    theta = _np.sum(upper_trianglar_sums) / _np.sum(event_counts)
    mu = diagonal_sums / time_duration

    return (omega, theta, mu)
