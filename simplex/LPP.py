class LinearProgrammingProblem:
    def __init__(self, A, b, c, alternative=False):
        #         Input data validation
        assert type(A) == np.ndarray and type(b) == np.ndarray \
               and type(c) == np.ndarray, 'Invalid input data type! Numpy arrays required!'
        assert len(A.shape) == 2 and len(b.shape) == 1 and len(c.shape) == 1 \
               and A.shape[0] == b.shape[0] and A.shape[1] == c.shape[0], 'Shape(A) = shape(b) * shape(c)'

        self.A = A
        self.b = b
        self.alternative = alternative
        self.c = c
        print('Input data is OK!')
        #         Basis and free variable indices (s0 - constant -> indices + 1)
        self.basis = np.arange(self.c.shape[0]) + self.A.shape[0] + 1
        self.free = np.arange(self.c.shape[0]) + 1
        self.simplex_matrix = np.hstack((b.reshape(-1, 1), A))  # without minus!!!!! s - (...)
        self.simplex_matrix = np.vstack((self.simplex_matrix, np.array([0] + list(-c))))

    def target_function(self, x, inices, add=0):
        def sumproduct(*lists):
            return sum(functools.reduce(operator.mul, data) for data in zip(*lists))

        return sumproduct(x, self.c[inices]) + add

    def calculate_simplex_table(self, r, k):  # r - row, k - column variable to swap
        assert k > 0 and k < self.simplex_matrix.shape[1], 'Invalid variable number to swap'
        assert r >= 0 and r < self.simplex_matrix.shape[0], 'Invalid constraint number'

        #         NOT OPTIMAL !!!!
        for i in range(self.simplex_matrix.shape[0]):
            for j in range(self.simplex_matrix.shape[1]):
                if i != r and j != k:
                    self.simplex_matrix[i, j] -= self.simplex_matrix[i, k] * self.simplex_matrix[r, j] \
                                                 / self.simplex_matrix[r, k]

        for j in range(self.simplex_matrix.shape[1]):
            if j != k:
                self.simplex_matrix[r, j] /= self.simplex_matrix[r, k]

        for i in range(self.simplex_matrix.shape[0]):
            if i != r:
                self.simplex_matrix[i, k] /= -self.simplex_matrix[r, k]
        #         NOT OPTIMAL !!!!

        self.simplex_matrix[r, k] = 1 / self.simplex_matrix[r, k]
        old_basis = self.basis[r]
        self.basis[r] = self.free[k - 1]
        self.free[k - 1] = old_basis

    def find_descision_indices_step_1(self):
        less_zero = np.where(self.simplex_matrix[:-1, 0] < 0)[0]
        try:
            i_0 = less_zero[0]
        except IndexError:
            print('Available solution is found!')
            print('Basis variables: ', self.basis)
            print('Free variables: ', self.free)
            return None, None
        less_zero = self.simplex_matrix[i_0, 1:]

        if functools.reduce(operator.and_, less_zero >= 0, True):
            raise ValueError('No available solution')

        k = np.where(less_zero < 0)[0][0] + 1

        fractions = self.simplex_matrix[:-1, 0] / self.simplex_matrix[:-1, k]
        r = np.where(fractions > 0, fractions, np.inf).argmin()
        return r, k

    def find_descision_indices_step_2(self):
        less_zero = np.where(self.simplex_matrix[-1:, 1:].flatten() < 0)[0]
        try:
            j_0 = less_zero[0] + 1
        except IndexError:
            print('Optimal soulution is found!')
            print('Basis variables: ', self.basis)
            print('Free variables: ', self.free)
            if self.alternative:
                print('Optimal value: ', -self.simplex_matrix[-1, 0])
            else:
                print('Optimal value: ', self.simplex_matrix[-1, 0])
            return None, None
        descision_column = self.simplex_matrix[:-1, j_0]
        if functools.reduce(operator.and_, descision_column <= 0, True):
            raise ValueError('Function is unlimited!')

        k = j_0
        fractions = self.simplex_matrix[:-1, 0] / descision_column
        r = np.where(fractions > 0, fractions, np.inf).argmin()
        return r, k