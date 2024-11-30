import numpy as np
from math import sqrt


def f(x):  #calculates f at given x
  f_at_x = np.sin((x - 2)**2)  # define function here
  return f_at_x


def f1(x, h):  #calculates f' at given x using the five point stencil method
  z = (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)
  return (z)


def f2(x, b):  #calculates f'' at given x using the five point stencil method with f' values given by f1
  a = (-f1(x + 2 * b, b) + 8 * f1(x + b, b) - 8 * f1(x - b, b) +
       f1(x - 2 * b, b)) / (12 * b)
  return (a)

def find_min(x1, a, b, c, epsilon):
  x2 = 0
  beta = 0.000001  #precision for f1,f2

  while (abs(x1 - x2) > epsilon):  #b_n variables control the flow
    # Lines marked with "#" are for debugging purposes
    b4 = 0
    b3 = 0
    
#Step1
    if (f2(x1, beta) < -0.000001):  
      #print('step1')
      b4 = 1
#Step2
    if (b4 == 0):  
      #print('step2')
      y1 = x1 - (f1(x1, beta) / f2(x1, beta))
      tester1 = f(x1) - 0.5 * c * ((f1(x1, beta))**2 / f2(x1, beta))
      if (a < y1 < b or f(y1) < tester1):
        x2 = y1
      else:
        b3 = 1
#Step3
    if (b3 == 1):  
      #print('step3')
      i = 0
      looper = 1
      while (looper == 1):
        yi = x1 - f1(x1, beta) / pow(2, i)
        tester3 = -c * (-f1(x1, beta)**2 / (pow(2, i)))
        if (a < yi < b and f(yi) - f(x1) < tester3):
          looper = 0

        #print('yi = ', yi, sep='')
        #print('tester condition = ', f(yi) - f(x1) - tester3, sep='')  # must be <0
        i = i + 1
        #print('itr3 ', i, sep='')
      x2 = yi
#Step4
    if (b4 == 1):  
      #print('step4')
      dk = 1
      i = 0
      looper = 1
      if (f1(x1, beta) > 0):
        dk = -1
      while (looper == 1):
        yi = x1 - f1(x1, beta) / pow(2, i) + dk / (sqrt(pow(2, i)))
        tester4 = c * (-f1(x1, beta)**2 / (pow(2, i))) + f2(x1, beta) / (pow(2, i + 1))
        if ((a < yi < b and f(yi) - f(x1) <= tester4)):
          looper = 0
        #print('yi = ', yi, sep='')
        #print('tester condition = ', f(yi) - f(x1) - tester4, sep='')  # must be <0
        i = i + 1
        #print('itr4 ', i, sep='')
        if (i > 150):
          print('overflow')
          break
      x2 = yi

    temp = x1
    x1 = x2
    x2 = temp
  return(x1)

"""Searching for all local minima, in (a,b), by successive approximations of x1"""
def main():
  print('function can be changed in line 6')  
  print('insert interval (a,b)')  
  a = float(input('a = '))
  b = float(input('b = '))
  c = float(input('insert 0< c <1:\n'))
  d = int(input('insert density for local minima search, for example 10,20,50, integer:\n'))
  epsilon = int(input('insert accuracy for method, for example 100,2000,10000, integer:\n'))
  rounding = len(str(epsilon))  #rounding numbers depending on wanted accuracy
  
  epsilon = 1.0/ epsilon
  h = (b - a) / d  #step size
  Roots = [b-1]
  
  for i in range(d-1):  
    x = round(a + (i+1) * h, rounding)  #x is the approximation of the root
    root = find_min(x, a, b, c, epsilon)  #finding a minimum for each x
    
    if (round(root, rounding) in Roots): #Checking if root already exists in list
      pass
      #print('passed')
    else:
      """Checking if root is a or b, Not needed if looking for roots in [a,b]"""
      if(abs(root-a) > epsilon and abs(root-b) > epsilon):  
        Roots.append(round(root, rounding))
  
  
  Roots.pop(0)
  print('Local minima in (', a, ',', b, ') are:', sep='')
  print(Roots)
    

if __name__ == "__main__":
  main()
