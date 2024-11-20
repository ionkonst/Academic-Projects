import numpy as np
import random as rd
import copy as cp
import time
#program 2 is gray.py
#import from gray.py binary2gray, gray2binPopulation
from gray import binary2gray, gray2binPopulation

G = 10  #total generations
N = 8  #number of chromosomes per generation. Even number
L = 9  #binary length
C_MINMAX_FIT_PERCENTAGE = 0.01
Cross_prob = 0.8  #probability of crossover
Mutation_prob = 0.01 #probability of mutation
a = -15  #domain [a,b]
b = 15
GrayEncoding = 0  #0 for binary, 1 for gray
rd.seed(1)


#calculation of a function f(x)
def f(x):
  f_at_x = 2 * np.sin(x) + x + 0.1 * pow(x - 5.5, 2)  # define function here
  return f_at_x


#Population initialization
def init_population():
  Population = []
  for c in range(N):
    chromosome = ''
    for j in range(L):
      bit = str(rd.randint(0, 1))
      chromosome = chromosome + bit
    Population.append(chromosome)
  return Population


#converts binary string list to integer list
def binStr2int_List(pop):
  inpt = cp.deepcopy(pop)
  for i in range(N):
    inpt[i] = int(inpt[i], 2)
  return inpt


#converts binary string list to float list in [a, b]
def intRemap_List(pop):
  inpt = cp.deepcopy(pop)
  inpt = binStr2int_List(inpt)
  nom = b - a
  den = pow(2, L) - 1
  frac = nom / den
  for i in range(N):
    inpt[i] = a + frac * inpt[i]
  return inpt


#calculates fitness of each chromosome, input float list
def fitness(pop):
  inpt = cp.deepcopy(pop)
  inpt = intRemap_List(inpt)
  for i in range(N):
    inpt[i] = f(inpt[i])
  return inpt


#compute average fitness of population
def avg_fitness(pop):
  inpt = cp.deepcopy(pop)
  inpt = fitness(inpt)
  avg = np.mean(inpt)
  return avg


#compute diversity of the population by summing the hamming distance of each chromosome
def normalized_ham_diversity(pop):
  inpt = cp.deepcopy(pop)
  sum_distances = 0
  for i in range(N):
    for j in range(i + 1, N):
      distance = 0
      for k in range(L):
        if inpt[i][k] != inpt[j][k]:
          distance += 1
      sum_distances += distance / L
  norm = 0.5 * N * (N - 1)
  norm_sum_distances = sum_distances / norm
  return norm_sum_distances


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


#crossover over 2 binary strings
def crossover(bin1, bin2):
  foo = rd.random()
  if (foo < Cross_prob):
    cutpoint = rd.randint(1, L - 1)
    #print(cutpoint)
    child1 = bin1[0:cutpoint] + bin2[cutpoint:L]
    child2 = bin2[0:cutpoint] + bin1[cutpoint:L]
    bin1 = child1
    bin2 = child2
  return [bin1, bin2]


#mutation of a binary string
def mutation(bin):
  bar = list(bin)
  for i in range(L):
    foo = rd.random()
    if (foo < Mutation_prob):
      bar[i] = str(1 - int(bar[i]))
  bin = "".join(bar)

  return bin


#combination of selection, crsossover and mutation over a bin string population
def evolve1generation(pop):
  inpt = cp.deepcopy(pop)
  sel_chroms = []
  nextgen = []
  for i in range(N):
    j = selection(inpt)  #select individuals
    if (GrayEncoding == 1):
      sel_chroms.append(binary2gray(inpt[j]))
    else:
      sel_chroms.append(inpt[j])
    if (i % 2 == 1):
      next2 = crossover(sel_chroms[i - 1], sel_chroms[i])  #crossover
      for j in range(2):
        mutated = mutation(next2[j])  #mutation
        nextgen.append(mutated)
  if (GrayEncoding == 1):
    nextgen = gray2binPopulation(nextgen)
  return (nextgen)


def example1():
  print('change algorithm parameters in line 5')
  print('function can be changed in line 18')

  Population = init_population()
  for i in range(G - 1):
    print('--Generation: ', i, '--', sep='')
    x_points = intRemap_List(Population)
    #print([ '%.2f' % elem for elem in x_points ])
    print('Average Fitness: ', avg_fitness(Population))
    print('Diversity: ', normalized_ham_diversity(Population))
    print('Best Solution: ', min(fitness(Population)))
    Population = evolve1generation(Population)
  print('--Last Generation: ', G - 1, '--', sep='')
  print(Population)
  print('Average Fitness: ', avg_fitness(Population))
  print('Diversity: ', normalized_ham_diversity(Population))
  print("Best Solution: ", min(fitness(Population)))


def example2():
  G_list = [5, 10, 15, 25, 50]  #total generations
  N_list = [4, 8, 12, 16,
            20]  #number of chromosomes per generation. Even number
  L_list = [5, 6, 9, 12, 15, 20]  #binary length
  C_prob_list = [1, 0.8, 0.7, 0.5, 0.3]  #probability of crossover
  M_prob_list = [0, 0.01, 0.05, 0.1, 0.3]  #probability of mutation
  #average results for number of reps
  reps = 1000
  for x in L_list:
    global L
    L = x
    avg = gen = div = min_fit = 0
    start_time = time.time()
    for j in range(reps):
      rd.seed(88 * j + 2024)
      Population = init_population()
      a = avg_fitness(Population)
      d = normalized_ham_diversity(Population)
      m = min(fitness(Population))
      g = 0
      #find best fitness and avg fitness, diversity, generation when achieved
      for i in range(G - 1):
        Population = evolve1generation(Population)
        if (min(fitness(Population)) < m):
          g = i
          #print(i)
          a = avg_fitness(Population)
          d = normalized_ham_diversity(Population)
          m = min(fitness(Population))
      #averaging results
      avg = avg + a
      div = div + d
      min_fit = min_fit + m
      gen = gen + g
      res = [
          round(min_fit / reps, 2),
          round(avg / reps, 2),
          round(div / reps, 2),
          round(gen / reps, 2)
      ]
      end_time = time.time()
      elapsed_time = round(end_time - start_time, 2)
    print(G, N, L, Cross_prob, Mutation_prob, ":", res, "in", elapsed_time,
          "sec")


def example3():
  G_list = [5, 10, 15, 25, 50]  #total generations
  N_list = [4, 8, 12, 16, 20]  #number of chromosomes per generation. Even number
  L_list = [6, 9, 12, 15, 20]  #binary length
  C_prob_list = [1, 0.8, 0.7, 0.5, 0.3]  #probability of crossover
  M_prob_list = [0, 0.01, 0.05, 0.1, 0.3]  #probability of mutation
  #average results for number of reps
  reps = 10
  Results = []
  Parameters = []
  for G in G_list:
    print("G=", G, "***")
    for n in N_list:
      global N
      N = n
      print("N=", N, "**")
      for l in L_list:
        global L
        L = l
        print("L=", L)
        for c in C_prob_list:
          global Cross_prob
          Cross_prob = c
          for m in M_prob_list:
            global Mutation_prob
            Mutation_prob = m
            avg = gen = div = min_fit = 0
            for j in range(reps):
              rd.seed(520 * j + 68)
              Population = init_population()
              a = avg_fitness(Population)
              d = normalized_ham_diversity(Population)
              m = min(fitness(Population))
              g = 0
              #find best fitness and avg fitness, diversity, generation when achieved
              for i in range(G - 1):
                Population = evolve1generation(Population)
                if (min(fitness(Population)) < m):
                  g = i
                  #print(i)
                  a = avg_fitness(Population)
                  d = normalized_ham_diversity(Population)
                  m = min(fitness(Population))
              #averaging results
              avg = avg + a
              div = div + d
              min_fit = min_fit + m
              gen = gen + g
              res = [
                  round(min_fit / reps, 2),
                  round(avg / reps, 2),
                  round(div / reps, 2),
                  round(gen / reps, 2)
              ]
            Results.append(res)
            foo = [G, N, L, Cross_prob, Mutation_prob]
            Parameters.append(foo)
  #List of lists of GA parameters
  print(Parameters)
  #List of lists of GA metrics. Parameters[i] corresponds to Results[i]
  print(Results)
  print(len(Parameters), "elements")


def main():
  example2()
   
