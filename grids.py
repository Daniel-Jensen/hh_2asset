
import os
import numpy as np
import scipy.linalg
from sequence_jacobian import grids


_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


print("DATA DIR:", _DATA_DIR)
print("Trying to load:", os.path.join(_DATA_DIR, "Px_GMAR.txt"))

# Make grids for b, a, k, and z
# For either one or two countries

def make_grids_1(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):

    b_grid = grids.agrid(amax=bmax, n=nB)
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()

    if nZ == 19:

        markov_ctstime = np.loadtxt(os.path.join(_DATA_DIR, "Px_GMAR.txt"))
        e_grid = np.loadtxt(os.path.join(_DATA_DIR, "x_vec.txt")).flatten()

        # Continuous-time → discrete-time
        markov_distime = scipy.linalg.expm(markov_ctstime)

        # Row normalize
        row_sums = markov_distime.sum(axis=1)
        Pi = markov_distime / row_sums[:, None]

    else:
        e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    return b_grid, a_grid, k_grid, e_grid, Pi



def make_grids_2(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z, lam, Y, zeta):

    b_grid = grids.agrid(amax=bmax, n=nB)
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()

    if nZ == 19:

        markov_ctstime = np.loadtxt(os.path.join(_DATA_DIR, "Px_GMAR.txt"))
        e_grid = np.loadtxt(os.path.join(_DATA_DIR, "x_vec.txt")).flatten()

        # Continuous-time → discrete-time
        markov_distime = scipy.linalg.expm(markov_ctstime)

        # Row normalize
        row_sums = markov_distime.sum(axis=1)
        Pi = markov_distime / row_sums[:, None]

    else:
        e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    return b_grid, a_grid, k_grid, e_grid, Pi
