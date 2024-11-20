# Based on the code from user mits. Source:
# https://www.geeksforgeeks.org/gray-to-binary-and-binary-to-gray-conversion/
def xor_c(a, b):
  return '0' if (a == b) else '1'


def flip(c):
  return '1' if (c == '0') else '0'


def binary2gray(binary):
  gray = ""
  gray += binary[0]
  for i in range(1, len(binary)):
    gray += xor_c(binary[i - 1], binary[i])
  return gray


def gray2binary(gray):
  binary = ""
  binary += gray[0]
  for i in range(1, len(gray)):
    if (gray[i] == '0'):
      binary += binary[i - 1]
    else:
      binary += flip(binary[i - 1])
  return binary

def bin2grayPopulation(pop):
  grayPop = []
  for i in range(len(pop)):
    grayPop.append(binary2gray(pop[i]))
  return grayPop

def gray2binPopulation(grayPop):
  binPop = []
  for i in range(len(grayPop)):
    binPop.append(gray2binary(grayPop[i]))
  return binPop
