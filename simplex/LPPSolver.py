import numpy as np
class LPPSolver:
    def __init__(self, problem, verbose=True):
        self.problem = problem
        self.verbose = verbose

    def get_solution(self):
        basis_inices = np.where(self.problem.basis <= self.problem.c.shape[0])[0]
        if len(basis_inices) == 0:
            return 0
        basis_names = self.problem.basis[basis_inices] - 1
        target_vector = self.problem.simplex_matrix[basis_inices, 0]
        if self.problem.alternative:
            return -self.problem.target_function(target_vector, basis_names)
        return self.problem.target_function(target_vector, basis_names)
    
    def get_optimal_vector(self):
        basis_inices = np.where(self.problem.basis <= self.problem.c.shape[0])[0]
        if len(basis_inices) == 0:
            return 0
        basis_names = self.problem.basis[basis_inices] - 1      
        target_vector = np.zeros(self.problem.c.shape[0])
        target_vector[basis_names] = self.problem.simplex_matrix[basis_inices, 0]
        return target_vector
    
    def step_1(self):
        iteration = 1
        print('Searching available solution...')
        r, k = self.problem.find_descision_indices_step_1()
        while r is not None:
            if self.verbose:
                print('Iteration: ', iteration)
            iteration += 1
            self.problem.calculate_simplex_table(r, k)
            if self.verbose:
                print('==========================')
                print(self.problem.simplex_matrix)
                print('==========================')
            r, k = self.problem.find_descision_indices_step_1()
        if self.verbose:
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
            if self.verbose:
                print('Iteration: ', iteration)
            iteration += 1
            self.problem.calculate_simplex_table(r, k)
            if self.verbose:
                print('==========================')
                print('Basis variables:', self.problem.basis)
                print('Free variables:', self.problem.free)
                print(self.problem.simplex_matrix)
                print('Current optimal solution: ', self.get_solution())
                print('==========================')
            r, k = self.problem.find_descision_indices_step_2()
        if self.verbose:
            print('Simplex matrix after Step 2')
            print(self.problem.simplex_matrix)

    def solve(self):
        self.step_1()
        self.step_2()