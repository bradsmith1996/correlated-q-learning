import cvxpy as cp
import numpy as np
from cvxopt import matrix, solvers

# A = np.ones([7,6]) # 44 linear inequalities, 26 variables
# A[5][0] = 0
# A[6][:]*= -1
# A[6][0] = 0
# A[0][:] = np.array([1,0,-2,-1,1,2],dtype=float)
# A[1][:] = np.array([1,2,0,-2,-1,1],dtype=float)
# A[2][:] = np.array([1,1,2,0,-2,-1],dtype=float)
# A[3][:] = np.array([1,-1,1,2,0,-2],dtype=float)
# A[4][:] = np.array([1,-2,-1,1,2,0],dtype=float)
# print(A)
# c = np.array([1,0,0,0,0,0], dtype=float)
# b = np.array([0,0,0,0,0,1,-1], dtype=float)
# print(c)
# print(b)
# x = cp.Variable(6)
# prob = cp.Problem(cp.Maximize(c.T @ x),
#    [A @ x <= b, x[1:]>=0.0])
# prob.solve()
# print(x.value)

# Set up c vector:
# There are 25 variables:
# Initialize the A matrix:
A = np.ones([12,6]) # 44 linear inequalities, 26 variables
A[5][0] = 0
A[6][:]*= -1
A[6][0] = 0
# Create the b vector:
# Testing  case with known results
A[0][:] = np.array([1,0,-2,-1,1,2],dtype=float)
A[1][:] = np.array([1,2,0,-2,-1,1],dtype=float)
A[2][:] = np.array([1,1,2,0,-2,-1],dtype=float)
A[3][:] = np.array([1,-1,1,2,0,-2],dtype=float)
A[4][:] = np.array([1,-2,-1,1,2,0],dtype=float)

A[0][1:] = 0
A[1][1:] = 0
A[2][1:] = 0
A[3][1:] = 0
A[4][1:] = 0

A[7][:] = np.array([0,-1,0,0,0,0],dtype=float)
A[8][:] = np.array([0,0,-1,0,0,0],dtype=float)
A[9][:] = np.array([0,0,0,-1,0,0],dtype=float)
A[10][:] = np.array([0,0,0,0,-1,0],dtype=float)
A[11][:] = np.array([0,0,0,0,0,-1],dtype=float)
A = matrix(A,tc='d')
c = matrix([-1,0,0,0,0,0],tc='d')
b = matrix([0,0,0,0,0,1,-1,0,0,0,0,0],tc='d')
print(A)
print(c)
print(b)
sol=solvers.lp(c,A,b)
print(sol['x'])