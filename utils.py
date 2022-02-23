import numpy as np


def generate_pscm(p, ratio, pr_edge, pr_exo):
    """
    Generate a linear P-SCM with given parameters.
        :param p: Number of observed variables.
        :param ratio: Source to observed node ratio.
        :param pr_edge: Probability of a causal connection between two observed variables
                    (i.e., average node degree/(p-1) ).
        :param pr_exo: Probability of an exogenous connection between an observed variable and a source variable
                    (i.e., average source degree/p).
        :return: Adjacency matrix A, exogenous connection matrix B, mixing matrix W.
    """
    m_max = int(np.round(p * ratio))

    # Generate adjacency matrix A
    check = False
    while not check:
        A = np.random.choice([0, 1], size=(p, p), p=[1 - pr_edge, pr_edge])
        A[np.triu_indices(p)] = 0

        # Generate exogenous connection matrix B
        B = np.random.choice([0, 1], size=(p, m_max), p=[1 - pr_exo, pr_exo])

        idx = ~B.any(axis=1)
        check = all(A[idx].any(axis=1))  # Check if there is variable with no connection

    A = (np.random.random(A.shape) / 2 + 0.5) * A
    A = A * np.random.choice([-1, 1], size=A.shape)
    B = B[:, np.any(B, axis=0)]  # Delete empty columns
    B = (np.random.random(B.shape) / 2 + 0.5) * B
    B = B * np.random.choice([-1, 1], size=B.shape)

    W = np.linalg.solve(np.eye(A.shape[0]) - A, B)

    per_row = np.random.permutation(A.shape[0])
    per_col = np.random.permutation(B.shape[1])

    A = A[np.ix_(per_row, per_row)]
    B = B[np.ix_(per_row, per_col)]
    W = W[np.ix_(per_row, per_col)]

    return A, B, W


def generate_dsp_scm(p, ratio, pr_edge, pr_exo):
    """
    Generate a linear DS-P-SCM with given parameters.
        :param p: Number of observed variables.
        :param ratio: Source to observed node ratio.
        :param pr_edge: Probability of a causal connection between two observed variables
                    (i.e., average node degree/(p-1) ).
        :param pr_exo: Probability of an exogenous connection between an observed variable and a source variable
                    (i.e., average source degree/p).
        :return: Adjacency matrix A, exogenous connection matrix B, mixing matrix W.
    """
    m_max = int(np.round(p * ratio) - p)
    A = np.random.choice([0, 1], size=(p, p), p=[1 - pr_edge, pr_edge])
    A[np.triu_indices(p)] = 0

    B = np.random.choice([0, 1], size=(p, m_max), p=[1 - pr_exo, pr_exo])
    idx = ((B != 0).sum(axis=0)) > 1
    B = np.hstack((B[:, idx], np.eye(p)))

    A = (np.random.random(A.shape) / 2 + 0.5) * A
    A = A * np.random.choice([-1, 1], size=A.shape)

    B = (np.random.random(B.shape) / 2 + 0.5) * B
    B = B * np.random.choice([-1, 1], size=B.shape)

    W = np.linalg.solve(np.eye(A.shape[0]) - A, B)

    per_row = np.random.permutation(A.shape[0])
    per_col = np.random.permutation(B.shape[1])

    A = A[np.ix_(per_row, per_row)]
    B = B[np.ix_(per_row, per_col)]
    W = W[np.ix_(per_row, per_col)]

    return A, B, W


def generate_dsp_scm_without_latent(p, pr_edge):
    """
    Generate a linear DS-P-SCM without latent confounders.
        :param p: Number of observed variables.
        :param pr_edge: Probability of a causal connection between two observed variables.
        :return: Adjacency matrix A, exogenous connection matrix B=I, mixing matrix W.
    """
    A = np.random.choice([0, 1], size=(p, p), p=[1 - pr_edge, pr_edge])
    A[np.triu_indices(p)] = 0
    A = (np.random.random(A.shape) / 2 + 0.5) * A
    A[abs(A) < 1e-3] = 0
    A = A * np.random.choice([-1, 1], size=A.shape)

    W = np.linalg.inv(np.eye(A.shape[0]) - A)
    W[abs(W) < 1e-3] = 0
    per_row = np.random.permutation(p)
    per_col = np.random.permutation(p)

    A = A[np.ix_(per_row, per_row)]
    B = np.eye(p)
    W = W[np.ix_(per_row, per_col)]

    return A, B, W