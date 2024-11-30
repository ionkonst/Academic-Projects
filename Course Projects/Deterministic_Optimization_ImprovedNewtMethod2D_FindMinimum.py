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

def main():
  print('function can be changed in line 6') 
  print('insert interval (a,b)')  #test values(0.25, 3.5)
  a = float(input('a = '))
  b = float(input('b = '))
  c = float(input('insert 0< c <1:\n'))
  epsilon = float(input('insert epsilon for method accuracy:\n'))

  '''initial guess for x'''
  x1 = float(input('insert initial guess for x:\n'))
  
  x2 = 0
  beta = 0.000001  #precision for f1,f2
  j = 0  #Total iterations 
  
  while (abs(x1 - x2) > epsilon):  #b_n variables control the flow
    # Lines marked with "#" are for debugging purposes
    b4 = 0
    b3 = 0
    
    if (f2(x1, beta) < -0.000001):  #Step1
      #print('step1')
      b4 = 1

    
    if (b4 == 0):  #Step2
      #print('step2')
      y1 = x1 - (f1(x1, beta) / f2(x1, beta))
      tester1 = f(x1) - 0.5 * c * ((f1(x1, beta))**2 / f2(x1, beta))
      if (a < y1 < b or f(y1) < tester1):
        x2 = y1
      else:
        b3 = 1

  
    if (b3 == 1):  #Step3
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
      

    if (b4 == 1):  #Step4
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

    j = j + 1
    if (j == 1):
      print("x0 = \n", x1, sep='')  #Print initial guess for x
    print("x", j, " = ", sep='')  
    temp = x1
    x1 = x2
    x2 = temp
    print(x1)
    
  print('one minimum is: ', round(x1,4))


if __name__ == "__main__":
  main()
