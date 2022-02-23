import numpy as np
from PSCM import PSCMRecovery
from utils import generate_dsp_scm_without_latent
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
from scipy.stats import t
from sklearn.utils import resample


def support(X, n, m, max_iter=50):
    """
    Use blind source separation to recover the mixing matrix.
        :param X: Observed samples.
        :param n: Number of samples in each bootstrapping iteration (usually 75% of the samples).
        :param m: Number of sources.
        :param max_iter: Number of iterations for bootstrapping.
        :return: Recovered mixing matrix W_learn.
    """

    X_sample = resample(X.T, n_samples=int(n), replace=False)
    transformer = FastICA(n_components=m).fit(X_sample)
    W_base = transformer.mixing_
    W_base = W_base / np.linalg.norm(W_base, axis=0)
    W_base = W_base / np.sign(W_base[np.argmax(abs(W_base), axis=0), range(m)])
    W_total = np.zeros((p, m, max_iter))
    W_total[:, :, 0] = W_base
    for ii in range(1, max_iter):
        X_sample = resample(X.T, n_samples=int(n), replace=False)
        transformer = FastICA(n_components=m).fit(X_sample)
        W_temp = transformer.mixing_
        W_temp = W_temp / np.linalg.norm(W_temp, axis=0)
        W_temp = W_temp / np.sign(W_temp[np.argmax(abs(W_temp), axis=0), range(m)])

        cost = np.zeros((m, m))
        for i in range(m):
            cost[i] = np.linalg.norm(W_base - W_temp[:, i][:, np.newaxis], axis=0) ** 2
        _, col = linear_sum_assignment(cost.T)
        W_total[:, :, ii] = W_temp[:, col]

    mu = np.mean(W_total, axis=2)
    std = np.std(W_total, axis=2)

    W_learn = mu * (abs(mu) > std * t.cdf(0.975, max_iter - 1) / max_iter ** 0.5 + 0.2)
    return W_learn


def successful_recovery(W_learned, W_actual):
    """
    Check if the BSS recovery is successful or not (i.e., the support is correctly recovered).
        :param W_learned: Recovered mixing matrix from BSS.
        :param W_actual: True mixing matrix.
        :return: Boolean value indicating whether the recovery is successful.
    """
    p, m = W_learned.shape
    W_learned = W_learned / abs(W_learned).max(axis=0)
    W_learned = W_learned / np.sign(W_learned[np.argmax(abs(W_learned), axis=0), range(m)])

    W_actual = W_actual / abs(W_actual).max(axis=0)
    W_actual = W_actual / np.sign(W_actual[np.argmax(abs(W_actual), axis=0), range(m)])

    cost = np.zeros((m, m))
    for i in range(m):
        cost[i] = np.linalg.norm(W_actual - W_learned[:, i][:, np.newaxis], axis=0) ** 2
    _, col = linear_sum_assignment(cost.T)
    W_learned = W_learned[:, col]
    diff = np.logical_xor(W_actual != 0, W_learned != 0)
    return not diff.any()


if __name__ == "__main__":
    # Initialization
    p = 5                   # Number observed variables
    pr_edge = 1.5 / (p-1)   # Average connection probability
    n = 1000                # Number of observed samples

    # Generate linear P-SCM that is uniquely identfiable and can be successfully recovered
    check_recovery = False
    while not check_recovery:
        # Generate linear P-SCM that is uniquely identifiable (i.e., satisfies the conditions)
        check_unique = False
        while not check_unique:
            A, B, W = generate_dsp_scm_without_latent(p=p, pr_edge=pr_edge)
            test = PSCMRecovery(a=A, b=B)
            check_unique = test.check_uniqueness()

        # Generate observed data with uniform noise
        m = W.shape[1]
        X = W.dot(np.random.uniform(size=(m, n)) - 0.5)

        # BSS recovery with bootstrapping
        W_learn = support(X=X, n=0.75*n, m=m, max_iter=40)

        # Check if the recovery is successful
        check_recovery = successful_recovery(W_learn, W)

    # Recover the model using PSCM Recovery Algorithm
    test = PSCMRecovery(w=W_learn, alpha=0.1)
    A_learn, B_learn = test.find_structure()

    print('True Adjacency Matrix:\n', A)
    print('Recovered Adjacency Matrix:\n', A_learn)
    print('Recovered Exogenous Connection Matrix:\n', B_learn)
