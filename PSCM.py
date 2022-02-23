import numpy as np


class PSCMRecovery:
    def __init__(self, w=None, a=None, b=None, alpha=1e-10):
        self.a_true = a    # True adjacency matrix (for checking satisfiability)
        self.b = b         # True exogenous connection matrix (for checking satisfiability)
        if w is None:
            self.w = np.linalg.solve(np.eye(a.shape[0]) - a, b)
        else:
            self.w = w

        self.p = self.w.shape[0]
        self.m = self.w.shape[1]
        self.alpha = alpha  # Threshold for pruning after recovery
        self.order = None
        self.rev_order = None
        self.unique = None

        self.pp = []        # Possible parent set
        self.comp = []      # Component set
        self.list_pp = []   # Possible parent set in list structure

        self.A = np.eye(self.p)  # Recovered adjacency matrix

    def permute(self):
        # Recover the correct causal order
        self.w[abs(self.w) < self.alpha] = 0
        nonzero_row = np.count_nonzero(self.w, axis=1)
        self.order = np.argsort(nonzero_row)
        self.rev_order = np.arange(len(self.order))[np.argsort(self.order)]
        self.w = self.w[self.order]
        if self.b is not None:
            self.b = self.b[self.order]
            self.a_true = self.a_true[np.ix_(self.order, self.order)]

    def find_pp(self):
        # Find possible parent set for each observed variable
        for k in range(self.p):
            self.comp.append(set(np.nonzero(self.w[k])[0]))
            temp_pp = {k}
            candidate_pp = set(range(k))

            for i in reversed(range(k)):
                if i in candidate_pp and self.comp[i] <= self.comp[k]:
                    temp_pp = temp_pp | self.pp[i]
                    candidate_pp = candidate_pp - self.pp[i]

            self.pp.append(temp_pp)

    def _find_unique_component(self, k, list_pp, check=False):
        # Find the unique component set of an observed variable, following the iteration procedure
        index = []
        uni = []
        for j in self.comp[k]:
            if check:
                temp_w = np.where(abs(self.b[list_pp, j]) > self.alpha)[0]
            else:
                temp_w = np.where(abs(self.w[list_pp, j]) > self.alpha)[0]

            if len(temp_w) == 1:
                i = list_pp[temp_w[0]]
                if i not in index:
                    index.append(i)
                    uni.append(j)
                else:
                    if check:
                        uni.append(j)

        return index, uni

    def _find_local_structure(self, k, list_pp):
        # Recover the local structure of the linear P-SCM at variable k
        index, uni = self._find_unique_component(k, list_pp)
        if index:

            while index:
                i = index.pop()
                j = uni.pop()

                self.A[k, i] = self.w[k, j] / self.w[i, j]
                self.w[k] = self.w[k] - self.A[k, i] * self.w[i]
                list_pp.remove(i)

            self._find_local_structure(k, list_pp)

        else:
            index_nz = np.where(np.any(abs(self.w[list_pp]) > self.alpha, axis=0))[0]
            if index_nz.any():
                solve = self.w[np.ix_(self.list_pp[k], index_nz)].T

                self.A[k, list_pp] = np.linalg.lstsq(solve, self.w[k, index_nz])[0]
            self.w[k, index_nz] = 0

    def find_structure(self):
        # Recover the structure of the linear P-SCM
        if self.unique is None:
            self.permute()
            self.find_pp()

        for k in range(self.p):
            list_pp = list(self.pp[k])
            list_pp.remove(k)

            self._find_local_structure(k, list_pp)
            self.w[k, abs(self.w[k]) < self.alpha] = 0

        self.A = np.eye(self.p) - np.linalg.inv(self.A)
        self.A[abs(self.A) < self.alpha] = 0
        return self.A[np.ix_(self.rev_order, self.rev_order)], self.w[self.rev_order]

    def check_uniqueness(self):
        # Check if the given linear P-SCM is uniquely identifiable
        self.permute()
        self.find_pp()

        self.unique = True

        for k in range(self.p):
            list_pp = list(self.pp[k])
            list_pp.remove(k)
            if self.a_true is not None:
                parent = set(np.where(abs(self.a_true[k]) > self.alpha)[0])
                if parent - set(list_pp):
                    self.unique = False
                    break

            self._check_local_uniqueness(k, list_pp)

            if not self.unique:
                break

        return self.unique

    def _check_local_uniqueness(self, k, list_pp, marriage_condition=False):
        # Check if variable k in the given linear P-SCM is uniquely identifiable
        index, uni = self._find_unique_component(k, list_pp, check=True)
        if index:
            list_pp = list(set(list_pp) - set(index))

            if any(abs(self.b[k, uni]) > self.alpha):
                self.unique = False
                return

            self._check_local_uniqueness(k, list_pp)
        else:
            n_nz = len(list_pp)
            index_nz = np.where(np.any(abs(self.b[list_pp]) > self.alpha, axis=0))[0]
            if marriage_condition and min(n_nz, len(index_nz)) > 0:
                # Check marriage condition by matrix rank
                cost_matrix = self.b[np.ix_(list_pp, index_nz)]
                check_mc = (np.linalg.matrix_rank(cost_matrix) < n_nz)
            else:
                # Comparing the number of elements in I_k and the components included
                check_mc = (len(index_nz) < n_nz)

            if check_mc or any(abs(self.b[k, index_nz]) > self.alpha):
                self.unique = False
