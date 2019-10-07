import numpy as np
from math import ceil, floor
from LPP import *
from LPPSolver import *
class IntegerLPPSolver:
    def __init__(self, solver):
        self.solver = solver
        self.problem = solver.problem
    
    def find_non_integer(self):
        optimal_vector = self.solver.get_optimal_vector()
        
        mask = optimal_vector != np.round(optimal_vector)
        var_id = np.where(mask)[0]
        if var_id.shape[0] == 0:
            return None, None
        var_id = var_id[0]
        var_value = optimal_vector[var_id]
        return var_id, var_value
    
    def create_new_problems(self):
        var_id, var_value = self.find_non_integer()
        if var_id is None:
            return -1, -1
        less_problem = None
        new_row = np.zeros((1, self.problem.A.shape[1]))
        new_row[0][var_id] = 1
        print('Creating less problem...') 
        less_problem_A = np.vstack((self.problem.A, new_row))
        less_problem_b = np.append(self.problem.b, floor(var_value))
        less_problem = LinearProgrammingProblem(less_problem_A, less_problem_b, self.problem.c)
        
        print('Creating greater problem...')
        greater_problem_A = np.vstack((self.problem.A, -new_row))
        greater_problem_b = np.append(self.problem.b, -ceil(var_value))
        greater_problem = LinearProgrammingProblem(greater_problem_A, greater_problem_b, self.problem.c)
        return less_problem, greater_problem
    
    def solve(self):
        new_problems = self.create_new_problems()
        if new_problems[0] == -1:
            return -1
        less_optimal = None
        less_optimal_vector = None
        greater_optimal = None
        greater_optimal_vector = None
        var_id, var_value = self.find_non_integer()
        if new_problems[0] is not None:
            less_solver = LPPSolver(new_problems[0], verbose=False)
            try:
                print('Solving less problem...')
                print('Additional constraint: x_' + str(var_id + 1) + ' <= ' + str(floor(var_value)))
                less_solver.solve()
                less_optimal = less_solver.get_solution()
                less_optimal_vector = less_solver.get_optimal_vector()
                print('======================')
            except ValueError:
                print('Less problem does not have any solution!')
        
        
        greater_solver = LPPSolver(new_problems[1], verbose=False)
        try:
            print('Solving greater problem...')
            print('Additional constraint: -x_' + str(var_id + 1) + ' <= ' + str(-ceil(var_value)))
            self.kek = new_problems[1]
            greater_solver.solve()
            greater_optimal = greater_solver.get_solution()
            greater_optimal_vector = greater_solver.get_optimal_vector()
            print('======================')
        except ValueError:
            print('Greater problem does not have any solution!')
            print('======================')
        if less_optimal is None and greater_optimal is None:
            raise ValueError('No integer solutions!')
        elif less_optimal is None or greater_optimal is not None and greater_optimal > less_optimal:
            print('Greater problem is better')
            self.problem = new_problems[1]
            self.solver = greater_solver
            self.optimal = greater_optimal
            self.optimal_vector = greater_optimal_vector
            print(self.optimal_vector)
            print('======================')
        elif greater_optimal is None or less_optimal is not None and less_optimal > greater_optimal:
            print('Less problem is better')
            self.problem = new_problems[0]
            self.solver = less_solver
            self.optimal = less_optimal
            self.optimal_vector = less_optimal_vector
            print(self.optimal_vector)
            print('======================')
        return 0
    
    def integer_solve(self):
        while self.solve() == 0:
            1
        print('Integer optimal solution is found!\n')
        print('New problem: (A, b)')
        print(self.problem.A)
        print(self.problem.b)
        print('\n')
        print('Integer optimal solution')
        print(self.solver.get_solution())
        print('Integer optimal vector')
        print(self.solver.get_optimal_vector())
        return