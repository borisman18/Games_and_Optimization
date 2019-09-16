import numpy as np
import operator
import functools
import warnings
warnings.filterwarnings('ignore')

A = np.array([
    [3, 1, 1],
    [1, 2, 0],
    [0, 0.5, 2]
]).astype(float)

b = np.array([3, 8, 1]).astype(float)
c = np.array([2, 6, 7]).astype(float)

print('A:')
print(A)
print('b: ', b)
print('c:', c)

class LinearProgrammingProblem:
    def __init__(self, A, b, c, opt_direction):
#         Input data validation
        assert type(A) == np.ndarray and type(b) == np.ndarray \
        and type(c) == np.ndarray, 'Invalid input data type! Numpy arrays required!'        
        assert len(A.shape) == 2 and len(b.shape) == 1 and len(c.shape) == 1 \
        and A.shape[0] == b.shape[0] and A.shape[1] == c.shape[0], 'Shape(A) = shape(b) * shape(c)'
        assert opt_direction in ('min', 'max'), 'Optimization direction must be \"min\" or \"max\"'
        
        self.A = A
        self.b = b
        if opt_direction == 'min':
            self.c = -c
        else:
            self.c = c
            
        self.opt_direction = 'max'
        print('Input data is OK!')
#         Basis and free variable indices (s0 - constant -> indices + 1)
        self.basis = np.arange(self.c.shape[0]) + self.A.shape[0] + 1
        self.free = np.arange(self.c.shape[0]) + 1
        self.simplex_matrix = np.hstack((b.reshape(-1,1), A)) # without minus!!!!! s - (...)
        self.simplex_matrix = np.vstack((self.simplex_matrix, np.array([0] + list(-c))))
        
    def target_function(self, x, inices, add=0):
        def sumproduct(*lists):
            return sum(functools.reduce(operator.mul, data) for data in zip(*lists))
        return sumproduct(x, self.c[inices]) + add
    
    def calculate_simplex_table(self, r, k): # r - row, k - column variable to swap
        assert k > 0 and k < self.simplex_matrix.shape[1], 'Invalid variable number to swap'
        assert r >= 0 and r < self.simplex_matrix.shape[0], 'Invalid constraint number'
        
#         NOT OPTIMAL !!!!
        for i in range(self.simplex_matrix.shape[0]):
            for j in range(self.simplex_matrix.shape[1]):
                if i != r and j != k:
                    self.simplex_matrix[i,j] -= self.simplex_matrix[i, k] * self.simplex_matrix[r, j] \
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
        self.basis[r] = self.free[k-1]
        self.free[k-1] = old_basis
        
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
            return None, None
        descision_column = self.simplex_matrix[:-1, j_0]
        if functools.reduce(operator.and_, descision_column <= 0, True):
            raise ValueError('Function is unlimited!')
            
        k = j_0
        fractions = self.simplex_matrix[:-1, 0] / descision_column
        r = np.where(fractions > 0, fractions, np.inf).argmin()
        return r, k

class LPPSolver:
    def __init__(self, problem, verbose=1):
        self.problem = problem
        self.verbose = verbose
        
    def get_solution(self):
        basis_inices = np.where(self.problem.basis <= self.problem.c.shape[0])[0]
        if len(basis_inices) == 0:
            return 0
        basis_names = self.problem.basis[basis_inices] - 1
        target_vector = self.problem.simplex_matrix[basis_inices, 0]
        return self.problem.target_function(target_vector, basis_names)
        
    def step_1(self):
        iteration = 1
        print('Searching available solution...')
        r, k = self.problem.find_descision_indices_step_1()
        while r is not None:
            print('Iteration: ', iteration)
            iteration += 1
            self.problem.calculate_simplex_table(r, k)
            print('==========================')
            print(self.problem.simplex_matrix)
            print('==========================')
            r, k = self.problem.find_descision_indices_step_1()
        print('Simplex matrix after Step 1')
        print(self.problem.simplex_matrix)
        print('Current basic solution: ', self.get_solution())
        print('\n')
        
    def step_2(self):
        iteration = 1
        print('Searching optimal solution...')
        r, k = self.problem.find_descision_indices_step_2()
        if r is None:
            print('Current optimal solution: ', self.get_solution())
        while r is not None:
            print('Iteration: ', iteration)
            iteration += 1
            self.problem.calculate_simplex_table(r, k)
            print('==========================')
            print('Basis variables:', self.problem.basis)
            print('Free variables:', self.problem.free)
            print(self.problem.simplex_matrix)
            print('Current optimal solution: ', self.get_solution())
            r, k = self.problem.find_descision_indices_step_2()
            print('==========================')
        print('Simplex matrix after Step 2')
        print(self.problem.simplex_matrix)
    def solve(self):
        self.step_1()
        self.step_2()
        
    
        


lpp = LinearProgrammingProblem(A, b, c, 'max')
solver = LPPSolver(lpp)        

solver.solve()