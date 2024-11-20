import numpy as np
import random as rd
import copy as cp
import time


G = 10  #total generations
N = 8  #number of chromosomes per generation. Even number
sigma = 1
C_MINMAX_FIT_PERCENTAGE = 0.01
Cross_prob = 0.8  #probability of crossover
Mutation_prob = 0.2 #probability of mutation
a = -15  #domain [a,b]
b = 15
rd.seed(3)

#calculation of a function f(x)
def f(x):
	f_at_x = 2 * np.sin(x) + x + 0.1 * pow(x - 5.5, 2)  # define function here
	return f_at_x

#Population initialization
def init_population():
	pop = []
	for i in range(N):
		pop.append(rd.uniform(a, b))
	return pop

#calculates fitness of each chromosome, input float list
def fitness(pop):
	inpt = cp.deepcopy(pop)
	for i in range(N):
		inpt[i] = f(inpt[i])
	return inpt

#compute average fitness of population
def avg_fitness(pop):
	inpt = cp.deepcopy(pop)
	inpt = fitness(inpt)
	avg = np.mean(inpt)
	return avg

#selects a chromosome for reproduction. Returns index of chromosome
def selection(pop):
	inpt = cp.deepcopy(pop)
	fit = fitness(inpt)
	c = C_MINMAX_FIT_PERCENTAGE * (max(fit) - min(fit))
	inv_fit = max(fit) + c - fit
	inv_fit_sum = sum(inv_fit)
	prob = []
	for i in range(N):
		if inv_fit_sum != 0:
			prob.append(inv_fit[i] / inv_fit_sum)
		else:
			#if i == 0:
			#print("total population convergence")
			prob.append(1 / N)

	i = N
	while (i >= N):  #in case of accumulation of float errors: repeat selection
		selector = rd.random()
		prob_sum = prob[0]
		i = 0
		#print(selector)
		#print(fit)
		#print(prob)
		#print(sum(prob))
		while (selector > prob_sum):  #selection loop
			prob_sum = prob_sum + prob[i]
			#print(prob_sum)
			i = i + 1
			#print(i)
	return i

#crossover via blending over 2
def blend(x1, x2):
	if (rd.random() < Cross_prob):
		b1 = rd.random()
		b2 = rd.random()
		y1 = b1 * x1 + (1 - b1) * x2
		y2 = b2 * x1 + (1 - b2) * x2
	else:
		y1 = x1
		y2 = x2
	return [y1, y2]

#mutation of a chromosome
def mutation(x):
	if (rd.random() < Mutation_prob):
		x = x + sigma*rd.uniform(-1, 1)
		if x > b or x < a:
			x = rd.uniform(a, b)
	return(x)

#combination of selection, crsossover and mutation over a bin string population
def evolve1generation(pop):
	inpt = cp.deepcopy(pop)
	sel_chroms = []
	nextgen = []
	for i in range(N):
		j = selection(inpt)  #select individuals
		sel_chroms.append(inpt[j])
		if (i % 2 == 1):
			next2 = blend(sel_chroms[i - 1], sel_chroms[i])  #crossover
			for j in range(2):
				mutated = mutation(next2[j])  #mutation
				nextgen.append(mutated)
	return (nextgen)

def example1():
	print('change algorithm parameters in line 5')
	print('function can be changed in line 18')

	Population = init_population()
	for i in range(G - 1):
		print('--Generation: ', i, '--', sep='')
		#print([ '%.2f' % elem for elem in x_points ])
		print('Average Fitness: ', avg_fitness(Population))
		print('Best Solution: ', min(fitness(Population)))
		Population = evolve1generation(Population)
	print('--Last Generation: ', G - 1, '--', sep='')
	print(Population)
	print('Average Fitness: ', avg_fitness(Population))
	print("Best Solution: ", min(fitness(Population)))

def example2():
	G_list = [5, 10, 15, 25, 50]  #total generations
	N_list = [4, 8, 12, 16,
						20]  #number of chromosomes per generation. Even number
	sigma_list = [0.1, 0.5, 1, 2, 3]  #mutation standard deviation
	C_prob_list = [1, 0.8, 0.7, 0.5, 0.3]  #probability of crossover
	M_prob_list = [0, 0.1, 0.2, 0.3, 0.5]  #probability of mutation
	#average results for number of reps
	reps = 100
	for x in M_prob_list:
		global Mutation_prob
		Mutation_prob = x
		avg = gen = div = min_fit = 0
		start_time = time.time()
		for j in range(reps):
			rd.seed(77 * j + 2024)
			Population = init_population()
			a = avg_fitness(Population)
			m = min(fitness(Population))
			g = 0
			#find best fitness and avg fitness, diversity, generation when achieved
			for i in range(G - 1):
				Population = evolve1generation(Population)
				if (min(fitness(Population)) < m):
					g = i
					#print(i)
					a = avg_fitness(Population)
					m = min(fitness(Population))
			#averaging results
			avg = avg + a
			min_fit = min_fit + m
			gen = gen + g
			res = [
					round(min_fit / reps, 2),
					round(avg / reps, 2),
					round(gen / reps, 2)
			]
			end_time = time.time()
			elapsed_time = round(end_time - start_time, 2)
		print(G, N, sigma, Cross_prob, Mutation_prob, ":", res, "in", elapsed_time,
					"sec")

def main():
	example2()

if __name__ == "__main__":
	main()
