from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

__all__ = ["eigenshuffle_eig", "eigenshuffle_eigh"]

# ---- Type aliases ------------------------------------------------------------
ArrayF = npt.NDArray[np.floating]  # real-valued arrays
ArrayC = npt.NDArray[np.complexfloating]  # complex-valued arrays
ArrayFC = ArrayF | ArrayC  # real or complex

EigenValsT = TypeVar("EigenValsT", ArrayF, ArrayC)
EigenVecsT = TypeVar("EigenVecsT", ArrayF, ArrayC)


def eigval_cost(vec1: ArrayF, vec2: ArrayF) -> ArrayF:
    """
    Compute the interpoint distance matrix between two sets of eigenvalues.

    Args:
        vec1 (npt.NDArray[np.floating]): First eigenvalue array.
        vec2 (npt.NDArray[np.floating]): Second eigenvalue array.

    Returns:
        npt.NDArray[np.floating]: Cost matrix of absolute differences between
        elements of vec1 and vec2, with shape (len(vec1), len(vec2)).
    """
    return np.abs(vec1[:, np.newaxis] - vec2[np.newaxis, :])


def _shuffle(
    eigenvalues: EigenValsT,
    eigenvectors: EigenVecsT,
    use_eigenvalues: bool = True,
) -> tuple[EigenValsT, EigenVecsT]:
    """
    Consistently reorder eigenvalues/vectors based on the initial ordering. Uses the
    Hungarian Algorithm (via scipy.optimize.linear_sum_assignment) to solve the
    assignment problem of which eigenvalue/vector pair most closely matches another.

    The distance function used here is:
        (1 - np.abs(V1.conj().T @ V2)) * np.sqrt(
            eigval_cost(D1.real, D2.real)**2
            + eigval_cost(D1.imag, D2.imag)**2
        )
    where eigval_cost computes the interpoint distance matrix and D, V are the
    eigenvalues/vectors, respectively.

    Args:
        eigenvalues (ArrayF | ArrayC): mxn eigenvalues
        eigenvectors (ArrayF | ArrayC): mxnxn eigenvectors
        use_eigenvalues (bool, optional): whether to include eigenvalue distances
            in the cost. Defaults to True.

    Returns:
        tuple[ArrayF | ArrayC, ArrayF | ArrayC]:
            consistently ordered eigenvalues/vectors.
    """
    for i in range(1, len(eigenvalues)):
        # compute distance between successive systems
        D1, D2 = eigenvalues[i - 1 : i + 1]
        V1, V2 = eigenvectors[i - 1 : i + 1]

        distance = 1 - np.abs(V1.conj().T @ V2)

        if use_eigenvalues:
            dist_vals = np.sqrt(
                eigval_cost(D1.real, D2.real) ** 2 + eigval_cost(D1.imag, D2.imag) ** 2
            )
            distance *= dist_vals

        # Hungarian assignment: rows = previous, cols = current
        row_ind, col_ind = linear_sum_assignment(distance)
        # For a square cost, row_ind should be [0..n-1]
        eigenvectors[i] = V2[:, col_ind]
        eigenvalues[i] = D2[col_ind]

        # phase/sign alignment (real- and complex-safe)
        V_prev = eigenvectors[i - 1]
        V_curr = eigenvectors[i]
        overlaps = np.sum(V_prev.conj() * V_curr, axis=0)  # per-column ⟨v_prev|v_curr⟩
        tol = 1e-12

        if np.isrealobj(V_prev) and np.isrealobj(V_curr):
            # keep arrays real: just flip signs using the real overlap
            signs = np.where(overlaps.real < 0.0, -1.0, 1.0)
            eigenvectors[i] = V_curr * signs
        else:
            # complex phase alignment; avoid unstable rotation when |overlap|≈0
            denom = np.maximum(np.abs(overlaps), tol)
            phases = overlaps.conj() / denom
            eigenvectors[i] = V_curr * phases

    return eigenvalues, eigenvectors


def _reorder(
    eigenvalues: EigenValsT, eigenvectors: EigenVecsT
) -> tuple[EigenValsT, EigenVecsT]:
    """
    Reorder eigenvalues (mxn) and eigenvectors (mxnxn) for each i entry (m) from low
    to high.

    Args:
        eigenvalues (ArrayF | ArrayC): mxn eigenvalue array
        eigenvectors (ArrayF | ArrayC): mxnxn eigenvector array

    Returns:
        tuple[ArrayF | ArrayC, ArrayF | ArrayC]: reordered eigenvalues and eigenvectors
    """
    indices_sort_all = np.argsort(eigenvalues.real)  # always valid
    for i in range(len(eigenvalues)):
        # initial ordering is purely ascending by real part
        indices_sort = indices_sort_all[i]
        eigenvalues[i] = eigenvalues[i][indices_sort]
        eigenvectors[i] = eigenvectors[i][:, indices_sort]
    return eigenvalues, eigenvectors


def _eigenshuffle(
    matrices: Sequence[ArrayFC] | ArrayFC,
    hermitian: bool,
    use_eigenvalues: bool,
) -> tuple[ArrayFC, ArrayFC]:
    """
    Consistently reorder eigenvalues and eigenvectors based on the initial ordering,
    which sorts the eigenvalues from low to high, then uses assignment to maintain
    correspondence across the sequence.

    Args:
        matrices: sequence or stacked array of shape (m, n, n)
        hermitian: whether to use eigh (Hermitian) or eig (general)
        use_eigenvalues: include eigenvalue distances in the matching cost

    Returns:
        (eigenvalues, eigenvectors) with consistent ordering across m.
    """
    assert len(np.shape(matrices)) > 2, "matrices must be of shape mxnxn"

    if hermitian:
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrices)

    eigenvalues, eigenvectors = _reorder(eigenvalues, eigenvectors)
    eigenvalues, eigenvectors = _shuffle(eigenvalues, eigenvectors, use_eigenvalues)
    return eigenvalues, eigenvectors


def eigenshuffle_eigh(
    matrices: Sequence[ArrayFC] | ArrayFC,
    use_eigenvalues: bool = True,
) -> tuple[ArrayF, ArrayFC]:
    """
    Compute eigenvalues and eigenvectors with eigh (Hermitian) of a series of matrices
    (mxnxn) and keep eigenvalues and eigenvectors consistently sorted; starting with the
    lowest eigenvalue.

    Returns:
        (eigenvalues, eigenvectors), where eigenvalues are real-valued.
    """
    # eigh → real eigenvalues; eigenvectors may be real or complex depending on input
    eigvals, eigvecs = _eigenshuffle(
        matrices, hermitian=True, use_eigenvalues=use_eigenvalues
    )
    # mypy-friendly cast: eigvals are guaranteed real for Hermitian problems
    return eigvals.real, eigvecs


def eigenshuffle_eig(
    matrices: Sequence[ArrayFC] | ArrayFC,
    use_eigenvalues: bool = False,
) -> tuple[ArrayFC, ArrayFC]:
    """
    Compute eigenvalues and eigenvectors with eig of a series of matrices (mxnxn) and
    keep eigenvalues and eigenvectors consistently sorted; starting with the lowest
    eigenvalue.

    Note:
        Default `use_eigenvalues=False` here because for non-Hermitian problems the
        complex eigenvalue distances can be less informative for tracking.

    Returns:
        (eigenvalues, eigenvectors), both possibly complex-valued.
    """
    return _eigenshuffle(matrices, hermitian=False, use_eigenvalues=use_eigenvalues)
