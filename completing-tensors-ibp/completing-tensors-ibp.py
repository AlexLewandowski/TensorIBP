import sys
import cPickle as CP
import matplotlib.pyplot as plt

import numpy as NP
import scipy.io as SPIO

from utils import *
from scipy import misc
from PyIBP import PyIBP as IBP

# IBP parameter (gamma hyperparameters)
(alpha, alpha_a, alpha_b) = (1., 1., 1.)
# Observed data Gaussian noise (Gamma hyperparameters)
(sigma_x, sx_a, sx_b) = (1., 1., 1.)
# Latent feature weight Gaussian noise (Gamma hyperparameters)
(sigma_a, sa_a, sa_b) = (1., 1., 1.)

# Data
sample = NP.random.choice([True, False],  data[:, :, 0].shape, p=[0.3, 0.7])
miss = sample.astype(int)

datar = data[:, :, 0]
datag = data[:, :, 1]
datab = data[:, :, 2]

rfill = NP.zeros(datar.shape)
gfill = NP.zeros(datag.shape)
bfill = NP.zeros(datab.shape)


# IBP parameter (gamma hyperparameters)
(alpha, alpha_a, alpha_b) = (1., 1., 1.)
# Observed data Gaussian noise (Gamma hyperparameters)
(sigma_x, sx_a, sx_b) = (1., 1., 1.)
# Latent feature weight Gaussian noise (Gamma hyperparameters)
(sigma_a, sa_a, sa_b) = (1., 1., 1.)

# Number of full sampling sweeps
numsamp = 10
burn_in = 0

# Center the data
rcdata = IBP.centerData(datar)
gcdata = IBP.centerData(datag)
bcdata = IBP.centerData(datab)


# Initialize the model
r = IBP(rcdata, (alpha, alpha_a, alpha_b),
        (sigma_x, sx_a, sx_b),
        (sigma_a, sa_a, sa_b),
        missing=miss)
g = IBP(gcdata, (alpha, alpha_a, alpha_b),
        (sigma_x, sx_a, sx_b),
        (sigma_a, sa_a, sa_b),
        missing=miss)
b = IBP(bcdata, (alpha, alpha_a, alpha_b),
        (sigma_x, sx_a, sx_b),
        (sigma_a, sa_a, sa_b),
        missing=miss)


# Do inference
for s in range(numsamp):
  # Print current chain state
  r.sampleReport(s)
  r.fullSample()
  g.fullSample()
  b.fullSample()
  if s > burn_in:
    rfill[sample] = rfill[sample] + r.X[sample]
    gfill[sample] = gfill[sample] + g.X[sample]
    bfill[sample] = bfill[sample] + b.X[sample]


# Real Image
plt.imshow(data)
plt.show()


# Incomplete Image
nandata = NP.copy(will)
nandata[:, :, 0][sample] = 0
nandata[:, :, 1][sample] = 0
nandata[:, :, 2][sample] = 0
plt.imshow(nandata)
plt.show()


# Mean Substitution
olddata = im2double(data)
meandata = im2double(data)
rmean = NP.mean(meandata[:, :, 0])
gmean = NP.mean(meandata[:, :, 1])
bmean = NP.mean(meandata[:, :, 2])
meandata[:, :, 0][sample] = rmean
meandata[:, :, 1][sample] = gmean
meandata[:, :, 2][sample] = bmean
plt.imshow(meandata)
plt.show()
print(NP.mean((olddata[:, :, 0][sample] - rmean)**2))
print(NP.mean((olddata[:, :, 1][sample] - gmean)**2))
print(NP.mean((olddata[:, :, 2][sample] - bmean)**2))


# Column Mean Substitution
meandata = im2double(data)
olddata = im2double(data)
for i in range(will.shape[1]):
  rmean = NP.mean(meandata[:, i, 0])
  gmean = NP.mean(meandata[:, i, 1])
  bmean = NP.mean(meandata[:, i, 2])
  meandata[:, i, 0][sample[:, i]] = rmean
  meandata[:, i, 1][sample[:, i]] = gmean
  meandata[:, i, 2][sample[:, i]] = bmean
plt.imshow(meandata)
plt.show()
print(NP.mean((olddata[:, :, 0][sample] - meandata[:, :, 0][sample])**2))
print(NP.mean((olddata[:, :, 1][sample] - meandata[:, :, 1][sample])**2))
print(NP.mean((olddata[:, :, 2][sample] - meandata[:, :, 2][sample])**2))


# Row Mean Substitution
meandata = im2double(data)
olddata = im2double(data)
for i in range(will.shape[0]):
  rmean = NP.mean(meandata[i, :, 0])
  gmean = NP.mean(meandata[i, :, 1])
  bmean = NP.mean(meandata[i, :, 2])
  meandata[i, :, 0][sample[i, :]] = rmean
  meandata[i, :, 1][sample[i, :]] = gmean
  meandata[i, :, 2][sample[i, :]] = bmean
plt.imshow(meandata)
plt.show()
print(NP.mean((olddata[:, :, 0][sample] - meandata[:, :, 0][sample])**2))
print(NP.mean((olddata[:, :, 1][sample] - meandata[:, :, 1][sample])**2))
print(NP.mean((olddata[:, :, 2][sample] - meandata[:, :, 2][sample])**2))


# Completed Image
completed = NP.copy(data)
olddata = im2double(data)
completed[:, :, 0][sample] = r.featMeans(
    datar)[sample] + rfill[sample] / (numsamp - burn_in)
completed[:, :, 1][sample] = g.featMeans(
    datag)[sample] + gfill[sample] / (numsamp - burn_in)
completed[:, :, 2][sample] = b.featMeans(
    datab)[sample] + bfill[sample] / (numsamp - burn_in)
completed = im2double(completed)
plt.imshow(completed)
plt.show()
print(NP.mean((olddata[:, :, 0][sample] - completed[:, :, 0][sample])**2))
print(NP.mean((olddata[:, :, 1][sample] - completed[:, :, 1][sample])**2))
print(NP.mean((olddata[:, :, 2][sample] - completed[:, :, 2][sample])**2))


# The fill-ins seem noisy, but slices yield approximations to the image
completed = NP.zeros(data.shape)
completed[:, :, 0][sample] = rfill[sample] / (numsamp - burn_in)
completed[:, :, 1][sample] = gfill[sample] / (numsamp - burn_in)
completed[:, :, 2][sample] = bfill[sample] / (numsamp - burn_in)
plt.imshow(completed)
plt.show()


completed = NP.zeros((data.shape))
completed[:, :, 0][sample] = r.featMeans(
    datar)[sample] + rfill[sample] / (numsamp - burn_in)
completed[:, :, 1][sample] = g.featMeans(
    datag)[sample] + gfill[sample] / (numsamp - burn_in)
completed[:, :, 2][sample] = b.featMeans(
    datab)[sample] + bfill[sample] / (numsamp - burn_in)
plt.imshow(completed[:, :, 0])
plt.show()
plt.imshow(completed[:, :, 1])
plt.show()
plt.imshow(completed[:, :, 2])
plt.show()

sample = NP.random.choice([True, False],  data[:, :, 1].shape, p=[0.3, 0.7])
miss = sample.astype(int)

tensorsample = NP.zeros(data.shape)
for i in range(data.shape[2]):
  tensorsample[:, :, i] = sample

T = t.dtensor(data)
S = t.dtensor(tensorsample)

X1miss = S.unfold(0).astype(bool)
X2miss = S.unfold(1).astype(bool)
X3miss = S.unfold(2).astype(bool)

X1dat = T.unfold(0)
X2dat = T.unfold(1)
X3dat = T.unfold(2)

X1fill = NP.zeros(X1dat.shape)
X2fill = NP.zeros(X2dat.shape)
X3fill = NP.zeros(X3dat.shape)

# Number of full sampling sweeps
numsamp = 5
burn_in = 0

X1cdat = IBP.centerData(X1dat)
X2cdat = IBP.centerData(X2dat)
X3cdat = IBP.centerData(X3dat)

# Initialize the model
X1 = IBP(X1cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X1miss.astype(int))
X2 = IBP(X2cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X2miss.astype(int))
X3 = IBP(X3cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X3miss.astype(int))


# Do inference
for s in range(numsamp):
  # Print current chain state
  X3.sampleReport(s)
  X1.fullSample()
  X2.fullSample()
  X3.fullSample()
  if s > burn_in:
    X1fill[X1miss] = X1fill[X1miss] + X1.X[X1miss]
    X2fill[X2miss] = X2fill[X2miss] + X2.X[X2miss]
    X3fill[X3miss] = X3fill[X3miss] + X3.X[X3miss]


# Completed X1 Matrix
NewX1 = T.unfold(0)
OldX1 = im2double(X1dat)
NewX1[X1miss] = X1.featMeans(X1dat)[X1miss] + \
    X1fill[X1miss] / (numsamp - burn_in)
NewX1 = im2double(NewX1)
plt.imshow(NewX1.fold())
plt.show()
mean = NP.mean(NewX1)
print(NP.mean((OldX1[X1miss] - NewX1[X1miss])**2))
print(NP.mean((OldX1[X1miss] - mean)**2))
print((X1.X.size, (numsamp - burn_in) * (X1.ZV.size + X1.weights().size)))


# Completed X2 Matrix
NewX2 = T.unfold(1)
OldX2 = im2double(X2dat)
NewX2[X2miss] = X2.featMeans(X2dat)[X2miss] + \
    X2fill[X2miss] / (numsamp - burn_in)
NewX2 = im2double(NewX2)
mean = NP.mean(NewX2)
plt.imshow(NewX2.fold())
plt.show()
print(NP.mean((OldX2[X2miss] - NewX2[X2miss])**2))
print(NP.mean((OldX2[X2miss] - mean)**2))
print((X2.X.size, (numsamp - burn_in) * (X2.ZV.size + X2.weights().size)))


# Completed X3 Matrix
NewX3 = T.unfold(2)
OldX3 = im2double(X3dat)
NewX3[X3miss] = X3.featMeans(X3dat)[X3miss] + \
    X3fill[X3miss] / (numsamp - burn_in)
NewX3 = im2double(NewX3)
mean = NP.mean(NewX3)
plt.imshow(NewX3.fold())
plt.show()
print(NP.mean((OldX3[X3miss] - NewX3[X3miss])**2))
print(NP.mean((OldX3[X3miss] - mean)**2))
print((X3.X.size, (numsamp - burn_in) * (X3.ZV.size + X3.weights().size)))


ZT = t.dtensor(NP.zeros(data.shape))

NewX1 = ZT.unfold(0)
NewX2 = ZT.unfold(1)
NewX3 = ZT.unfold(2)

NewX1[X1miss] = X1fill[X1miss] / (numsamp - burn_in)
NewX2[X2miss] = X2fill[X2miss] / (numsamp - burn_in)
NewX3[X3miss] = X3fill[X3miss] / (numsamp - burn_in)

plt.imshow(NewX1.fold())
plt.show()
plt.imshow(NewX1)
plt.show()
plt.imshow(NewX2.fold())
plt.show()
plt.imshow(NewX2)
plt.show()
plt.imshow(NewX3.fold())
plt.show()
plt.imshow(NewX3)
plt.show()


sample = NP.random.choice([True, False],  data[:, :, 1].shape, p=[0.3, 0.7])
miss = sample.astype(int)

tensorsample = NP.zeros(data.shape)
for i in range(data.shape[2]):
  tensorsample[:, :, i] = sample

T = t.dtensor(data)
S = t.dtensor(tensorsample)

X1miss = S.unfold(0).astype(bool)
X2miss = S.unfold(1).astype(bool)
X3miss = S.unfold(2).astype(bool)

X1dat = T.unfold(0)
X2dat = T.unfold(1)
X3dat = T.unfold(2)

X1fill = NP.zeros(X1dat.shape)
X2fill = NP.zeros(X2dat.shape)
X3fill = NP.zeros(X3dat.shape)

# Number of full sampling sweeps
numsamp = 5
burn_in = 0

X1cdat = IBP.centerData(X1dat)
X2cdat = IBP.centerData(X2dat)
X3cdat = IBP.centerData(X3dat)

# Initialize the model
X1 = IBP(X1cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X1miss.astype(int))
X2 = IBP(X2cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X2miss.astype(int))
X3 = IBP(X3cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X3miss.astype(int))


# Do inference
for s in range(numsamp):
  # Print current chain state
  X3.sampleReport(s)
  X1.fullSample()
  X2.fullSample()
  X3.fullSample()
  if s > burn_in:
    X1fill[X1miss] = X1fill[X1miss] + X1.X[X1miss]
    X2fill[X2miss] = X2fill[X2miss] + X2.X[X2miss]
    X3fill[X3miss] = X3fill[X3miss] + X3.X[X3miss]


N = 5
I = 30
J = 30
K = 40
A = NP.random.randint(-2, 3, (30, 5))
B = NP.random.randint(-2, 3, (30, 5))
C = NP.random.randint(-2, 3, (40, 5))

F = NP.zeros((30, 30, 40))
for n in range(N):
  for i in range(30):
    for j in range(30):
      for k in range(40):
        F[i, j, k] = F[i, j, k] + A[i, n] * \
            B[j, n] * C[k, n] + NP.random.normal(0, 5)


data = F
sample = NP.random.choice([True, False],  data.shape, p=[0.3, 0.7])

T = t.dtensor(data)
S = t.dtensor(sample)

X1miss = S.unfold(0).astype(bool)
X2miss = S.unfold(1).astype(bool)
X3miss = S.unfold(2).astype(bool)

X1dat = T.unfold(0)
X2dat = T.unfold(1)
X3dat = T.unfold(2)


X1fill = NP.zeros(X1dat.shape)
X2fill = NP.zeros(X2dat.shape)
X3fill = NP.zeros(X3dat.shape)

# Number of full sampling sweeps
numsamp = 35
burn_in = 30

X1cdat = IBP.centerData(X1dat)
X2cdat = IBP.centerData(X2dat)
X3cdat = IBP.centerData(X3dat)

# Initialize the model
X1 = IBP(X1cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X1miss.astype(int))
X2 = IBP(X2cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X2miss.astype(int))
X3 = IBP(X3cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X3miss.astype(int))


# Do inference
for s in range(numsamp):
  # Print current chain state
  X3.sampleReport(s)
  X1.fullSample()
  X2.fullSample()
  X3.fullSample()
  if s > burn_in:
    X1fill[X1miss] = X1fill[X1miss] + X1.X[X1miss]
    X2fill[X2miss] = X2fill[X2miss] + X2.X[X2miss]
    X3fill[X3miss] = X3fill[X3miss] + X3.X[X3miss]


# Completed X1 Matrix
NewX1 = T.unfold(0)
OldX1 = NP.copy(X1cdat)
NewX1[X1miss] = X1fill[X1miss] / (numsamp - burn_in)
mean = NP.mean(OldX1)
print(NP.mean((OldX1[X1miss] - NewX1[X1miss])**2))
print(NP.mean((OldX1[X1miss] - mean)**2))
print((X1.X.size, (numsamp - burn_in) * (X1.ZV.size + X1.weights().size)))


# Completed X2 Matrix
NewX2 = T.unfold(1)
OldX2 = NP.copy(X2cdat)
NewX2[X2miss] = X2fill[X2miss] / (numsamp - burn_in)
mean = NP.mean(OldX2)
print(NP.mean((OldX2[X2miss] - NewX2[X2miss])**2))
print(NP.mean((OldX2[X2miss] - mean)**2))
print((X2.X.size, (numsamp - burn_in) * (X2.ZV.size + X2.weights().size)))


# Completed X3 Matrix
NewX3 = T.unfold(2)
OldX3 = NP.copy(X3cdat)
NewX3[X3miss] = X3fill[X3miss] / (numsamp - burn_in)
mean = NP.mean(OldX3)
print(NP.mean((OldX3[X3miss] - NewX3[X3miss])**2))
print(NP.mean((OldX3[X3miss] - mean)**2))
print((X3.X.size, (numsamp - burn_in) * (X3.ZV.size + X3.weights().size)))

sample = NP.random.choice([True, False],  data[:, :, 1].shape, p=[0.3, 0.7])
miss = sample.astype(int)

tensorsample = NP.zeros(data.shape)
for i in range(data.shape[2]):
  tensorsample[:, :, i] = sample

T = t.dtensor(data)
S = t.dtensor(tensorsample)

X1miss = S.unfold(0).astype(bool)
X2miss = S.unfold(1).astype(bool)
X3miss = S.unfold(2).astype(bool)

X1dat = T.unfold(0)
X2dat = T.unfold(1)
X3dat = T.unfold(2)

X1fill = NP.zeros(X1dat.shape)
X2fill = NP.zeros(X2dat.shape)
X3fill = NP.zeros(X3dat.shape)

# Number of full sampling sweeps
numsamp = 20
burn_in = 10

X1cdat = IBP.centerData(X1dat)
X2cdat = IBP.centerData(X2dat)
X3cdat = IBP.centerData(X3dat)

# Initialize the model
X1 = IBP(X1cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X1miss.astype(int))


# Do inference
for s in range(numsamp):
  # Print current chain state
  print s
  X1.fullSample()
  if s > burn_in:
    X1fill[X1miss] = X1fill[X1miss] + X1.X[X1miss]

temp = X1.X
temp[X1miss] = X1fill[X1miss] / (numsamp - burn_in)

X2 = IBP(temp.fold().unfold(1), (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X2miss.astype(int))

for s in range(numsamp):
  # Print current chain state
  print s
  X2.fullSample()
  if s > burn_in:
    X2fill[X2miss] = X2fill[X2miss] + X2.X[X2miss]

temp = X2.X
temp[X2miss] = X2fill[X2miss] / (numsamp - burn_in)

X3 = IBP(temp.fold().unfold(2), (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X3miss.astype(int))

for s in range(numsamp):
  # Print current chain state
  print s
  X3.fullSample()
  if s > burn_in:
    X3fill[X3miss] = X3fill[X3miss] + X3.X[X3miss]


ZT = t.dtensor(NP.zeros(data.shape))

NewX1 = ZT.unfold(0)
NewX2 = ZT.unfold(1)
NewX3 = ZT.unfold(2)

NewX1[X1miss] = X1fill[X1miss] / (numsamp - burn_in)
NewX2[X2miss] = X2fill[X2miss] / (numsamp - burn_in)
NewX3[X3miss] = X3fill[X3miss] / (numsamp - burn_in)

plt.imshow(NewX1.fold())
plt.show()
plt.imshow(NewX1)
plt.show()
plt.imshow(NewX2.fold())
plt.show()
plt.imshow(NewX2)
plt.show()
plt.imshow(NewX3.fold())
plt.show()
plt.imshow(NewX3)
plt.show()


temp = NP.zeros((290, 245, 3))
temp[tensorsample.astype(bool)] = im2double(will)[tensorsample.astype(bool)]
plt.imshow(temp[:, :, :])
plt.show()


ZT = t.dtensor(NP.zeros(data.shape))

NewX1 = ZT.unfold(0)
NewX2 = ZT.unfold(1)
NewX3 = ZT.unfold(2)

NewX1[X1miss] = X1fill[X1miss] / (numsamp - burn_in)
NewX2[X2miss] = X2fill[X2miss] / (numsamp - burn_in)
NewX3[X3miss] = X3fill[X3miss] / (numsamp - burn_in)

plt.imshow(NewX1.fold() / 3 + NewX2.fold() / 3 + NewX3.fold() / 3)
plt.show()


A = NP.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = NP.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def outer(A, B, C):
  N = A.shape[1]
  I = A.shape[0]
  J = B.shape[0]
  K = C.shape[0]
  temp = NP.zeros((I, J, K))
  for n in range(N):
    a = A[:, n]
    b = B[:, n]
    c = C[:, n]
    temp += (a[:, None] * b)[:, :, None] * c
  return temp


def outer2(A, B):
  N = A.shape[1]
  I = A.shape[0]
  J = B.shape[0]
  temp = NP.zeros((I, J))
  for n in range(N):
    a = A[:, n]
    b = B[:, n]
    temp += (a[:, None] * b)
  return temp

sample = NP.random.choice([True, False], data[:, :, 1].shape, p=[0.3, 0.7])
miss = sample.astype(int)

tensorsample = NP.zeros(data.shape)
for i in range(data.shape[2]):
  tensorsample[:, :, i] = sample

T = t.dtensor(data)
S = t.dtensor(tensorsample)

X1miss = S.unfold(0).astype(bool)
X2miss = S.unfold(1).astype(bool)
X3miss = S.unfold(2).astype(bool)

X1dat = T.unfold(0)
X2dat = T.unfold(1)
X3dat = T.unfold(2)

X1fill = NP.zeros(X1dat.shape)
X2fill = NP.zeros(X2dat.shape)
X3fill = NP.zeros(X3dat.shape)

# Number of full sampling sweeps
numsamp = 20
burn_in = 10

X1cdat = IBP.centerData(X1dat)
X2cdat = IBP.centerData(X2dat)
X3cdat = IBP.centerData(X3dat)

# Initialize the model
X1 = IBP(X1cdat, (alpha, alpha_a, alpha_b),
         (sigma_x, sx_a, sx_b),
         (sigma_a, sa_a, sa_b),
         missing=X1miss.astype(int))


# Do inference
for s in range(numsamp):
  # Print current chain state
  print s
  X1.fullSample()
  if s > burn_in:
    X1fill[X1miss] = X1fill[X1miss] + X1.X[X1miss]
