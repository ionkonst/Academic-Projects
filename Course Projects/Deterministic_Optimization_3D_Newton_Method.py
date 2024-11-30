import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def f(x, y):  #calculates f at given x,y
	f = -x * x - y * y + x * y + x * x * x * x + y * y * y * y  # define function here!
	return f


def f_vec(X):
	return (f(X[0], X[1]))


def dx(x, y,
			 h):  #calculates f' at given x,y using the five point stencil method
	z = (-f(x + 2 * h, y) + 8 * f(x + h, y) - 8 * f(x - h, y) +
			 f(x - 2 * h, y)) / (12 * h)
	return (z)


def dy(x, y,
			 h):  #calculates f' at given x,y using the five point stencil method
	z = (-f(x, y + 2 * h) + 8 * f(x, y + h) - 8 * f(x, y - h) +
			 f(x, y - 2 * h)) / (12 * h)
	return (z)


def J(x, y, h):
	foo = [dx(x, y, h), dy(x, y, h)]
	return foo


def dxdx(x, y, b):
	a = -(2 * f(x, y) - f(x + b, y) - f(x - b, y)) / (b * b)
	return (a)


def dydy(x, y, b):
	a = -(2 * f(x, y) - f(x, y + b) - f(x, y - b)) / (b * b)
	return (a)


def dxdy(x, y, b):
	a = (f(x + b, y + b) + f(x - b, y - b) - f(x - b, y + b) -
			 f(x + b, y - b)) / (4 * b * b)
	return (a)


def dydx(x, y, b):
	a = dxdy(x, y, b)
	return (a)


def H(x, y, b):
	Hess = np.array([[dxdx(x, y, b), dxdy(x, y, b)],
									 [dydx(x, y, b), dydy(x, y, b)]])
	return Hess


def InverseMatrix(A):
	A_inv = np.linalg.inv(A)
	A_inv = np.ndarray.round(A_inv, decimals=5)
	return A_inv


def is_pos_def(x):
	return np.all(np.linalg.eigvals(x) > 0)


def eigenvec_from_Min_eigenval(A):
	w, v = np.linalg.eig(A)
	min_eig = min(w)
	min_index = w.tolist().index(min_eig)
	vec = v[:, min_index]  #eigenvector from min eigenvalue
	return vec


def sspiv(X, nabla, s, Domain, c):
	D = Domain
	control = True
	i = 0
	while (control):
		lk = (1.0 / pow(2, i))
		expr1 = X + lk * s
		arg1 = D[0] < expr1[0] < D[1] and D[2] < expr1[1] < D[3]
		expr2 = f_vec(X + lk * s) - f_vec(X)
		arg2 = expr2 <= c * lk * np.dot(s, nabla)
		if (not arg1 or not arg2):
			i += 1
		if (arg1 and arg2):
			control = False
	return (np.array(X + lk * s))


def sspv(X, nabla, nabla2, s, dk, Domain, c):
	D = Domain
	control = True
	i = 0
	while (control):
		denom = pow(2, i)
		lk = (1.0 / denom)
		expr1 = X + lk * s + (1.0 / pow(2, i / 2)) * dk
		arg1 = (D[0] < expr1[0] < D[1] and D[2] < expr1[1] < D[3])
		expr2 = f_vec(expr1) - f_vec(X)
		arg2 = (expr2 <= c * lk * (np.dot(s, nabla) + 0.5 * (dk @ nabla2 @ dk)))
		if (not arg1 or not arg2):
			i += 1
			if (i > 100): break  ###########
		if (arg1 and arg2):
			control = False
	return (np.array(X + lk * s + (1.0 / pow(2, i / 2)) * dk))


def find_min(X, Domain, c, epsilon):
	X1 = np.array([0, 0])
	h = 0.0000001  #precision for calculations f,dx,dy,dxdy,etc.
	j = 0
	while (np.sqrt(np.dot((X - X1), (X - X1))) > epsilon):
		j += 1
		#if (j>5): break ######

		#step1
		nabla = J(X[0], X[1], h)
		nabla2 = H(X[0], X[1], h)

		#step2
		if is_pos_def(nabla2):
			s0 = -InverseMatrix(nabla2) @ nabla
			X1 = sspiv(X, nabla, s0, Domain, c)

		#step3
		if not is_pos_def(nabla2):
			s0 = -np.array(nabla)
			s0 = s0 / np.linalg.norm(s0)
			d1 = eigenvec_from_Min_eigenval(nabla2)
			i = 1
			while (np.dot(d1, nabla) > 0):  #checking other eigenvectors
				d1 = i * ((-1)**i) * d1
				i += 1
				if (i > 1000):
					print("unable to find descending d0 with non positive curvature")
					break
			X1 = sspv(X, nabla, nabla2, s0, d1, Domain, c)

		temp = X
		X = X1
		X1 = temp
	#print("# of steps:",j)
	return X


def main():

	print('function can be changed in line 6')
	print('insert intervals (a1,b1)x(a2,b2) in Domain')
	a1 = float(input('a1 = '))
	b1 = float(input('b1 = '))
	a2 = float(input('a2 = '))
	b2 = float(input('b2 = '))
	c = float(input('insert 0< c <1:\n'))
	dx = int(
			input(
					'insert x-axis density for local minima search, for example 10,20,50, integer:\n'
			))
	dy = int(
			input(
					'insert y-axis density for local minima search, for example 10,20,50, integer:\n'
			))
	epsilon = int(
			input(
					'insert accuracy for method, for example 100,2000,10000, integer:\n')
	)

	epsilon = 1.0 / epsilon
	Domain = [a1, b1, a2, b2]

	X = [0.2, -0.2]
	#c = 0.8
	#epsilon = 0.001
	h1 = (b1 - a1) / dx  #step size x
	h2 = (b2 - a2) / dy  #step size y

	Roots = []
	Points = []

	for i in range(dx - 1):
		for j in range(dy - 1):
			X = [a1 + (i + 1) * h1, a2 + (j + 1) * h2]
			if (X != [0, 0]):
				sol = find_min(X, Domain, c, epsilon)
				print(sol)
				SpacePoint = np.append(sol, f_vec(sol))
				Roots.append(sol)
				Points.append(SpacePoint)

	#clustering

	amount_initial_centers = 1
	initial_centers = kmeans_plusplus_initializer(
			Roots, amount_initial_centers).initialize()
	# Create instance of X-Means algorithm. The algorithm will start analysis from 1 cluster
	# max number of clusters is 10
	xmeans_instance = xmeans(Roots, initial_centers, 10)
	xmeans_instance.process()
	# Extract clustering results: clusters and their centers
	clusters = xmeans_instance.get_clusters()
#print("clusters")
	centers =xmeans_instance.get_centers()

	print("local minima at (x,y):",centers)


	#plotting

	#plot the solutions
	x, y = zip(*Roots)
	# Plotting the points
	plt.scatter(x, y, color='blue', marker='o', label='2D Points')
	# Adding labels and title
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.title('2D Points Plot')
	# Display the legend
	plt.legend()
	# Show the plot
	plt.show()


if __name__ == "__main__":
	main()
