import numpy as np
from scipy.linalg import block_diag
import random


def constant_diagonal_matrix(D, const_var):
    return const_var * np.eye(D)


def diagonal_matrix(var):
    return np.diag(var)


def linear_var(D):
    return diagonal_matrix(np.arange(1, D + 1))


def exponential_var(D):
    return np.exp(linear_var(D))


def block_random_diag(D, n):
    """
    D : dimension
    n : number of random blocks
    """
    # generate blocks of random sizes whose sum is D
    blocks = random_sum_pos_int(n, D)
    return block_diagonal(blocks)


def block_diagonal(block_dim):
    """
    Generate block diagonal covariance matrices with given dimensionality.

    Parameters:
    n_blocks (int): Number of blocks in the matrix.
    block_dim (list of int): Dimensionality of each block.

    Returns:
    ndarray: Block diagonal covariance matrix.
    """

    cov_blocks = []
    for k in block_dim:
        cov_blocks.append(random_covariance(k))
    cov_matrix = block_diag(*cov_blocks)
    return (cov_matrix + cov_matrix.T) / 2  # Make symmetric by averaging with its transpose


def toeplitz_cov(dim, alpha):
    """
    Generate a semi-definite positive symmetric Toeplitz covariance matrix with given dimensionality and decay factor.

    Parameters:
    dim (int): Dimensionality of the matrix.
    alpha (float): Decay factor for the Toeplitz matrix.

    Returns:
    ndarray: Toeplitz covariance matrix.
    """
    # Construct the first row of the Toeplitz matrix
    row = np.zeros(dim)
    row[0] = 1
    for i in range(1, dim):
        row[i] = alpha ** i

    # Construct the matrix using the first row and its transpose
    toeplitz_matrix = np.zeros((dim, dim))
    for i in range(dim):
        toeplitz_matrix[i, i:] = row[:dim - i]
        toeplitz_matrix[i + 1:, i] = row[1:dim - i]

    # Make the matrix symmetric by averaging with its transpose

    # Make the matrix semi-definite positive by adding a multiple of the identity matrix
    toeplitz_matrix = toeplitz_matrix + np.eye(dim) * np.abs(np.min(np.linalg.eigvalsh(toeplitz_matrix)))

    return toeplitz_matrix


def hetero_cov(dim, var):
    # Define vector of variances
    # variances = np.arange(1, dim+1)
    # variances = variances
    # variances = variances ** 2

    # Create diagonal matrix
    Diag = np.diag(var)

    # Define correlation matrix
    C = random_covariance(dim)

    # Multiply diagonal and correlation matrices
    covariance_matrix = np.matmul(np.matmul(Diag**0.5, C), Diag**0.5)

    return covariance_matrix


def random_covariance(D):
    # Random covariance matrix
    # Generate a random matrix with uniformly distributed values between -1 and 1
    random_cov = np.random.rand(D, D) * 2 - 1

    # Make the matrix symmetric by averaging it with its transpose
    random_cov = (random_cov + random_cov.T) / 2

    # Add some positive diagonal values to ensure it is positive definite
    random_cov += np.eye(D) * D
    return random_cov


def is_symmetric_positive_definite(matrix):
    """
    Check if a matrix is symmetric and semi-definite positive.

    Parameters:
    matrix (ndarray): Matrix to be checked.

    Returns:
    bool: True if the matrix is symmetric and semi-definite positive, False otherwise.
    """
    if not np.allclose(matrix, matrix.T):
        # Check if the matrix is symmetric
        return False

    eigenvalues, _ = np.linalg.eig(matrix)
    if not np.all(eigenvalues > 0):
        # Check if all eigenvalues are positive
        return False

    return True


def random_sum_pos_int(n, N):
    """
    Generate n positive random integers whose sum is N.

    Parameters:
    n (int): Number of random integers to generate.
    N (int): Sum of the random integers.

    Returns:
    list: List of n positive random integers whose sum is N.
    """
    # Generate n-1 random numbers between 1 and N-1
    rand_nums = [random.randint(0, N-n) for _ in range(n - 1)]

    # Sort the random numbers in ascending order
    rand_nums.sort()

    # Compute the differences between adjacent random numbers
    diffs = []
    diffs.append(rand_nums[0] + 1)
    for i in range(0, n - 2):
        diffs.append(rand_nums[i + 1] - rand_nums[i] + 1)
    diffs.append(N - n - rand_nums[-1] + 1)

    # Append the remaining integer to make the sum exactly equal to N

    # Return the list of random integers
    return diffs


if __name__ == '__main__':
    a = hetero_cov(100, np.exp(np.arange(1,101)))
    print(a)
    print(1/np.diag(np.linalg.inv(a)))
    print(is_symmetric_positive_definite(a))
