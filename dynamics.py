#!/usr/bin/env python3
"""this python file contains the implementation of the code of the paper
'Evolved interactions stabilize many coexisting phases in multicomponent fluids'

The modules contains a few global constants, which set parameters of the algorithm as
described in the paper. They typically do not need to be changed. A good entry point
into the code might be to create a random interaction matrix and a random initial
composition using `random_interaction_matrix` and `get_uniform_random_composition`,
respectively. The function `evolve_dynamics` can then be used to evolve Eq. 4 in the
paper to its stationary state, whose composition the function returns. The returned
composition matrix can be fed into `count_phases` to obtain the number of distinct
phases. An ensemble average over initial conditions is demonstrated in the function
`estimate_performance`, which also uses Eq. 5 of the paper to estimate how well the
particular interaction matrix obtains a given target number of phases. Finally,
`run_evolution` demonstrates the evolutionary optimization over multiple generations.
"""

from typing import List, Tuple

import numpy as np
from numba import njit
from scipy import cluster, spatial


DT_INITIAL: float = 1.0  # initial time step for the relaxation dynamics
TRACKER_INTERVAL: float = 10.0  # interval for convergence check
TOLERANCE: float = 1e-15  # tolerance used to decide when stationary state is reached

CLUSTER_DISTANCE: float = 1e-2  # cutoff value for determining composition clusters

PERFORMANCE_TOLERANCE: float = 0.5  # tolerance used when calculating performance
KILL_FRACTION: float = 0.3  # fraction of population that is replaced each generation

REPETITIONS: int = 64  # number of samples used to estimate the performance
ALPHA: float = 1e2  # parameter for the evolution rate

def random_interaction_matrix(
    num_comp: int, chi_mean: float = None, chi_std: float = 1
) -> np.ndarray:
    """create a random interaction matrix

    Args:
        num_comp (int): The component count
        chi_mean (float): The mean interaction strength
        chi_std (float): The standard deviation of the interactions

    Returns:
        The full, symmetric interaction matrix
    """
    if chi_mean is None:
        chi_mean = 3 + 0.4 * num_comp

    # initialize interaction matrix
    chis = np.zeros((num_comp, num_comp))

    # determine random entries
    num_entries = num_comp * (num_comp - 1) // 2
    chi_vals = np.random.normal(chi_mean, chi_std, num_entries)

    # build symmetric  matrix from this
    i, j = np.triu_indices(num_comp, 1)
    chis[i, j] = chi_vals
    chis[j, i] = chi_vals
    return chis


@njit
def get_uniform_random_composition(num_phases: int, num_comps: int) -> np.ndarray:
    """pick concentrations uniform from allowed simplex (sum of fractions < 1)

    Args:
        num_phases (int): the number of phases to pick concentrations for
        num_comps (int): the number of components to use

    Returns:
        The fractions of num_comps components in num_phases phases
    """
    phis = np.empty((num_phases, num_comps))
    for n in range(num_phases):
        phi_max = 1.0
        for d in range(num_comps):
            x = np.random.beta(1, num_comps - d) * phi_max
            phi_max -= x
            phis[n, d] = x
    return phis


@njit
def calc_diffs(phis: np.ndarray, chis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """calculates chemical potential and pressure

    Note that we only calculate the parts that matter in the difference

    Args:
        phis: The composition of all phases
        chis: The interaction matrix

    Returns:
        The chemical potentials and pressures in all phases
    """
    phi_sol = 1 - phis.sum()
    if phi_sol < 0:
        raise RuntimeError("Solvent has negative concentration")

    log_phi_sol = np.log(phi_sol)
    mu = np.log(phis)
    p = -log_phi_sol
    for i in range(len(phis)):  # iterate over components
        val = chis[i] @ phis
        mu[i] += val - log_phi_sol
        p += 0.5 * val * phis[i]
    # print(mu, p)
    return mu, p


@njit
def evolution_rate(phis: np.ndarray, chis: np.ndarray = None) -> np.ndarray:
    """calculates the evolution rate of a system with given interactions

    Args:
        phis: The composition of all phases
        chis: The interaction matrix

    Returns:
        The rate of change of the composition (Eq. 4)
    """
    num_phases, num_comps = phis.shape

    # get chemical potential and pressure for all components and phases
    mus = np.empty((num_phases, num_comps))
    ps = np.empty(num_phases)
    for n in range(num_phases):  # iterate over phases
        mu, p = calc_diffs(phis[n], chis)
        # print(mu, p)
        mus[n, :] = mu
        ps[n] = p
    # print("________________\n")
    # calculate rate of change of the composition in all phases
    dc = np.zeros((num_phases, num_comps))
    for n in range(num_phases):
        for m in range(num_phases):
            delta_p = ps[n] - ps[m]
            for i in range(num_comps):
                delta_mu = mus[m, i] - mus[n, i]
                dc[n, i] += phis[n, i] * (phis[m, i] * delta_mu - delta_p)
    return ALPHA*dc


@njit
def iterate_inner(phis: np.ndarray, chis: np.ndarray, dt: float, steps: int) -> None:
    """iterates a system with given interactions

    Args:
        phis: The composition of all phases
        chis: The interaction matrix
        dt (float): The time step
        steps (int): The step count
    """

    for _ in range(steps):
        # make a step
        phis += dt * evolution_rate(phis, chis)

        # check validity of the result
        if np.any(np.isnan(phis)):
            raise RuntimeError("Encountered NaN")
        elif np.any(phis <= 0):
            raise RuntimeError("Non-positive concentrations")
        elif np.any(phis.sum(axis=-1) <= 0):
            raise RuntimeError("Non-positive solvent concentrations")


def evolve_dynamics(chis: np.ndarray, phis_init: np.ndarray,tol = TOLERANCE) -> np.ndarray:
    """evolve a particular system governed by a specific interaction matrix

    Args:
        chis: The interaction matrix
        phis_init: The initial composition of all phases

    Returns:
        phis: The final composition of all phases
    """
    phis = phis_init.copy()
    
    phis_last = np.zeros_like(phis)

    dt = DT_INITIAL
    steps_inner = max(1, int(np.ceil(TRACKER_INTERVAL / dt)))

    # run until convergence
    while not np.allclose(phis, phis_last, rtol=tol, atol=tol):
        phis_last = phis.copy()

        # do the inner steps and reduce dt if necessary
        while True:
            try:
                iterate_inner(phis, chis, dt=dt, steps=steps_inner)
            except RuntimeError as err:
                # problems in the simulation => reduced dt and reset phis
                dt /= 2
                steps_inner *= 2
                phis[:] = phis_last

                if dt < 1e-7:
                    raise RuntimeError(f"{err}\nReached minimal time step.")
            else:
                break

    return phis

def modified_evolve_dynamics(chis: np.ndarray, phis_init: np.ndarray,tol = TOLERANCE) -> np.ndarray:
    """A modified version of the evolve_dynamics function 
    where the stationary state is evaluated by the convergence of the chemical potentials
    and pressures

    Args:
        chis: The interaction matrix
        phis_init: The initial composition of all phases

    Returns:
        phis: The final composition of all phases
    """
    phis = phis_init.copy()
    phis_last = np.zeros_like(phis)
    num_phases, num_comps = phis.shape
    mus = np.empty((num_phases, num_comps))
    ps = np.empty(num_phases)
    for n in range(num_phases):  # iterate over phases
        mu, p = calc_diffs(phis[n], chis)
        # print(mu, p)
        mus[n, :] = mu
        ps[n] = p
    mus_last = np.zeros_like(mus)
    ps_last = np.zeros_like(ps)
    dt = DT_INITIAL
    steps_inner = max(1, int(np.ceil(TRACKER_INTERVAL / dt)))

    # run until convergence
    while not np.allclose(mus, mus_last, rtol=tol, atol=tol):
        while not np.allclose(ps, ps_last, rtol=tol, atol=tol):
            phis_last = phis.copy()
            mus_last = mus.copy()
            ps_last = ps.copy()
            # do the inner steps and reduce dt if necessary
            while True:
                try:
                    iterate_inner(phis, chis, dt=dt, steps=steps_inner)
                except RuntimeError as err:
                    # problems in the simulation => reduced dt and reset phis
                    dt /= 2
                    steps_inner *= 2
                    phis[:] = phis_last

                    if dt < 1e-7:
                        raise RuntimeError(f"{err}\nReached minimal time step.")
                else:
                    break

    return phis


def count_phases(phis: np.ndarray) -> int:
    """calculate the number of distinct phases

    Args:
        phis: The composition of all phases

    Returns:
        int: The number of phases with distinct composition
    """
    # calculate distances between compositions
    dists = spatial.distance.pdist(phis)
    # obtain hierarchy structure
    links = cluster.hierarchy.linkage(dists, method="centroid")
    # flatten the hierarchy by clustering
    clusters = cluster.hierarchy.fcluster(links, CLUSTER_DISTANCE, criterion="distance")

    return int(clusters.max())

def collapse_phases(phis: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """ collapses phases with similar composition using clustering.

    Args:
        phis: The composition of all phases -- some of which are repeating
        tol: The tolerance for clustering
    returns:
        phis_out: The composition of all phases after collapsing.
    """
    dists = spatial.distance.pdist(phis)
    links = cluster.hierarchy.linkage(dists, method="centroid")
    clusters = cluster.hierarchy.fcluster(Z=links, t=tol, criterion="distance")
    # clusters = cluster.hierarchy.fcluster(links,, criterion="distance")
    n_clusters = len(set(clusters))

    phis_out = np.empty((n_clusters, phis.shape[1]))
    for i in set(clusters):
        phis_out[i-1] = np.mean(phis[clusters == i], axis=0)
    return phis_out