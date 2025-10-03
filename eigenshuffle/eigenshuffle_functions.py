from typing import Literal, Sequence, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

__all__ = ["eigenshuffle_eig", "eigenshuffle_eigh"]

eigenvals_complex_or_float = TypeVar(
    "eigenvals_complex_or_float",
    npt.NDArray[np.floating],
    npt.NDArray[np.complexfloating],
)

eigenvecs_complex_or_float = TypeVar(
    "eigenvecs_complex_or_float",
    npt.NDArray[np.floating],
    npt.NDArray[np.complexfloating],
)


def eigval_cost(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
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
    eigenvalues: eigenvals_complex_or_float,
    eigenvectors: eigenvecs_complex_or_float,
    use_eigenvalues: bool = True,
) -> tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]:
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
        eigenvalues (eigenvals_complex_or_float): mxn eigenvalues
        eigenvectors (eigenvecs_complex_or_float): mxnxn eigenvectors
        use_eigenvalues (bool, optional): bool specifying use of eigenvalues in distance calculation. Defaults to True.

    Returns:
        tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]:
            consistently ordered eigenvalues/vectors.
    """
    for i in range(1, len(eigenvalues)):
        # compute distance between systems
        D1, D2 = eigenvalues[i - 1 : i + 1]
        V1, V2 = eigenvectors[i - 1 : i + 1]

        distance = 1 - np.abs(V1.conj().T @ V2)

        if use_eigenvalues:
            dist_vals = np.sqrt(
                eigval_cost(D1.real, D2.real) ** 2 + eigval_cost(D1.imag, D2.imag) ** 2
            )
            distance *= dist_vals

        # Solve the assignment: rows = previous states, cols = current states
        row_ind, col_ind = linear_sum_assignment(distance)
        # row_ind should be [0,1,...,n-1] for a square cost. If you want, assert this:
        # assert np.array_equal(row_ind, np.arange(distance.shape[0]))
        eigenvectors[i] = eigenvectors[i][:, col_ind]
        eigenvalues[i] = eigenvalues[i, col_ind]

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
    eigenvalues: eigenvals_complex_or_float, eigenvectors: eigenvecs_complex_or_float
) -> tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]:
    """
    Reorder eigenvalues (mxn) and eigenvectors (mxnxn) for each i entry (m) from low
    to high.

    Args:
        eigenvalues (eigenvals_complex_or_float): mxn eigenvalue array
        eigenvectors (eigenvecs_complex_or_float): mxnxn eigenvector array

    Returns:
        tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]: reordered eigenvalues and eigenvectors
    """
    indices_sort_all = np.argsort(eigenvalues.real)
    for i in range(len(eigenvalues)):
        # initial ordering is purely in decreasing order.
        # If any are complex, the sort is in terms of the
        # real part.
        indices_sort = indices_sort_all[i]

        eigenvalues[i] = eigenvalues[i][indices_sort]
        eigenvectors[i] = eigenvectors[i][:, indices_sort]
    return eigenvalues, eigenvectors


@overload
def _eigenshuffle(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    hermitian: Literal[True],
    use_eigenvalues: bool,
) -> tuple[
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]: ...


@overload
def _eigenshuffle(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    hermitian: Literal[False],
    use_eigenvalues: bool,
) -> tuple[
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]: ...


def _eigenshuffle(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    hermitian: bool,
    use_eigenvalues: bool,
) -> tuple[
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]:
    """
    Consistently reorder eigenvalues and eigenvectors based on the initial ordering,
    which sorts the eigenvalues from low to

    Args:
        matrices (Sequence[NDArray], NDArray): eigenvalue/vector problems
        hermitian (bool): bool specifying hermitian
        use_eigenvalues (bool): bool specifying use of eigenvalues for re-ordering in _shuffle

    Returns:
        tuple[NDArray, NDArray]: consistently ordered eigenvalues/vectors
    """
    assert len(np.shape(matrices)) > 2, "matrices must be of shape mxnxn"

    if hermitian:
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrices)

    eigenvalues, eigenvectors = _reorder(
        eigenvalues=eigenvalues, eigenvectors=eigenvectors
    )
    eigenvalues, eigenvectors = _shuffle(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        use_eigenvalues=use_eigenvalues,
    )
    return eigenvalues, eigenvectors


def eigenshuffle_eigh(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = True,
) -> tuple[
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]:
    """
    Compute eigenvalues and eigenvectors with eigh (hermitian) of a series of matrices
    (mxnxn) and keep eigenvalues and eigenvectors consistently sorted; starting with the
    lowest eigenvalue.

    Args:
        matrices (Sequence[npt.NDArray[np.floating]] | npt.NDArray[np.floating] | Sequence[npt.NDArray[np.complexfloating]] | npt.NDArray[np.complexfloating]): mxnxn array of eigenvalue problems
        use_eigenvalues (bool, optional): Use the distance between successive eigenvalues as part of the shuffling. Defaults to True.

    Returns:
        tuple[ npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], ]: sorted eigenvalues and eigenvectors
    """
    return _eigenshuffle(matrices, hermitian=True, use_eigenvalues=use_eigenvalues)


def eigenshuffle_eig(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = False,
) -> tuple[
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]:
    """
    Compute eigenvalues and eigenvectors with eig of a series of matrices (mxnxn) and
    keep eigenvalues and eigenvectors consistently sorted; starting with the lowest
    eigenvalue.

    Args:
        matrices (Sequence[npt.NDArray[np.floating]] | npt.NDArray[np.floating] | Sequence[npt.NDArray[np.complexfloating]] | npt.NDArray[np.complexfloating]): mxnxn array of eigenvalue problems
        use_eigenvalues (bool, optional): Use the distance between successive eigenvalues as part of the shuffling. Defaults to False.
            This default is different from `eigenshuffle_eigh` because for non-hermitian matrices, eigenvalues can be complex and their distance
            is less meaningful for sorting than for real eigenvalues from hermitian matrices.

    Returns:
        tuple[ npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], ]: sorted eigenvalues and eigenvectors
    """
    return _eigenshuffle(matrices, hermitian=False, use_eigenvalues=use_eigenvalues)
