from PSCM import PSCMRecovery
from utils import generate_pscm, generate_dsp_scm


def satisfiability(p=5, ratio=2.0, pr_edge=0.5, pr_exo=0.3, dsp_scm=True):
    """
    Count the number of attempts to generate a linear P-SCM that can be uniquely recovered
    (i.e., satisfies the derived conditions).
        :param p: Number of observed nodes
        :param ratio: Source to observed node ratio. (Number of sources / Number of observed variables.)
        :param pr_edge: Average node degree (expected number of causal connections of an observed variable).
        :param pr_exo: Average source degree (expected number of exogenous connections of an observed variable).
        :param dsp_scm: If the model is a DSP-SCM.
        :return: Number of attempts count, Generated linear P-SCM (A,B)
    """
    check = False
    count = 0

    if pr_edge >= 1:  # average degree for nodes
        pr_edge = pr_edge / (p-1)

    if pr_exo >= 1:  # average degree for sources
        pr_exo = pr_exo / p

    while not check:
        if dsp_scm:
            A, B, _ = generate_dsp_scm(p=p, ratio=ratio, pr_edge=pr_edge, pr_exo=pr_exo)
        else:
            A, B, _ = generate_pscm(p=p, ratio=ratio, pr_edge=pr_edge, pr_exo=pr_exo)

        test = PSCMRecovery(a=A, b=B)
        check = test.check_uniqueness()

        count += 1
    return count, A, B  # result


if __name__ == '__main__':
    # Initialization
    is_dsp_scm = False
    p = 5
    deg_edge = 1.5  # d_e
    deg_exo = 1.5  # d_o
    ratio = 1.2

    n, A, B = satisfiability(p=p, ratio=ratio, pr_edge=deg_edge, pr_exo=deg_exo, dsp_scm=is_dsp_scm)

    print('Number of attempts needed:', n)
    print('Matrix A: \n', A)
    print('Matrix B: \n', B)
