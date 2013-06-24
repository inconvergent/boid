#!/usr/bin/python
# -*- coding: utf-8 -*-

def main():

  import numpy as np
  import cairo, Image
  from time import time as time
  from operator import itemgetter
  from matplotlib import pyplot as plt
  import sys

  from scipy.spatial import Delaunay, distance

  colstack = np.column_stack
  pdist    = distance.pdist
  cdist    = distance.cdist
  cos      = np.cos
  sin      = np.sin
  pi       = np.pi
  arctan2  = np.arctan2
  sqrt     = np.sqrt
  square   = np.square
  int      = np.int
  zeros    = np.zeros
  array    = np.array
  rand     = np.random.random
  xand     = np.logical_and


  pii = 2.*pi
  C = 0.5
  ITTMAX = 1000

  N = 1000
  ONE = 1./N
  STP = ONE

  FUZZ = STP*0.1

  NUM = 100
  FLOCK_RAD = 0.1

  SEPARATION_PRI = 0.5
  ALIGNMENT_PRI = 0.3
  COHESION_PRI = 0.2

  MAXACC = STP

  ## CLASSES

  class Meta(object):

    c = [0]
    X = zeros(NUM,dtype=np.float)
    Y = zeros(NUM,dtype=np.float)
    DX = zeros(NUM,dtype=np.float)
    DY = zeros(NUM,dtype=np.float)

    SEPX = zeros(NUM,dtype=np.float)
    SEPY = zeros(NUM,dtype=np.float)
    ALIX = zeros(NUM,dtype=np.float)
    ALIY = zeros(NUM,dtype=np.float)
    COHX = zeros(NUM,dtype=np.float)
    COHX = zeros(NUM,dtype=np.float)

    A = zeros( (NUM,NUM), dtype=np.float )
    R = zeros( (NUM,NUM), dtype=np.float )

    F = zeros( (NUM,NUM), dtype=np.bool )

    def set_dist(self):

      self.R[:] = cdist(*[ colstack((self.X,self.Y)) ]*2 ) 

    #def set_phi(self):

      #for i in xrange(NUM):
        #dx = self.X[i] - self.X
        #dy = self.Y[i] - self.Y
        #self.A[i,:] = arctan2(dy,dx)

    def set_flock(self):

      for i in xrange(NUM):
        d = self.R[i,:]
        inflock = d < FLOCK_RAD
        inflock[i] = False
        self.F[i,:] = inflock[:]

    def iterate(self):

      alpha = rand( NUM ) *pi*2.
      self.DX[:] += cos( alpha )*FUZZ
      self.DY[:] += sin( alpha )*FUZZ

      self.X[:] += self.DX[:]
      self.Y[:] += self.DY[:]


  class Boid(Meta):

    def __init__(self, x,y,dx,dy):
      self.i = self.c[0]
      self.c[0] += 1

      self.X[self.i] = x
      self.Y[self.i] = y
      self.DX[self.i] = dy
      self.DY[self.i] = dx

    def separation(self):

      self.SEPX[self.i] = 0.
      self.SEPY[self.i] = 0.

      inflock = self.F[self.i,:]
      inflock[self.i] = False

      if inflock.sum() > 0:

        scale = 1./square( self.R[self.i,inflock] )

        dx = ( self.X[self.i] - self.X[inflock] ) * scale
        dy = ( self.Y[self.i] - self.Y[inflock] ) * scale

        self.SEPX[self.i] = +dx.sum()
        self.SEPY[self.i] = +dy.sum()

    def alignment(self):

      self.ALIX[self.i] = 0.
      self.ALIY[self.i] = 0.

      inflock = self.F[self.i,:]
      inflock[self.i] = False

      if inflock.sum() > 0:

        scale = 1./square( self.R[self.i,inflock] )

        dx = self.DX[inflock] * scale
        dy = self.DY[inflock] * scale

        self.ALIX[self.i] = dx.sum()
        self.ALIY[self.i] = dy.sum()

    def step(self):

      self.DX[self.i] = 0.
      self.DY[self.i] = 0.

      acc = sqrt( square( self.SEPX[self.i] ) + \
                  square( self.SEPY[self.i] ) )
      available = MAXACC
      if acc > available:
        scale = available / acc
        self.SEPX[self.i] = self.SEPX[self.i] * scale
        self.SEPY[self.i] = self.SEPY[self.i] * scale
        accsum = MAXACC
      else:
        accsum = acc

      self.DX[self.i] += self.SEPX[self.i]
      self.DY[self.i] += self.SEPY[self.i]

      if accsum >= MAXACC:
        return

      #acc = sqrt(square(self.ALIX[self.i]) + square(self.ALIY[self.i]))
      #available = MAXACC - accsum
      #if acc > available:
        #scale = available / acc
        #self.ALIX[self.i] = self.ALIX[self.i] * scale
        #self.ALIY[self.i] = self.ALIY[self.i] * scale

      #self.DX[self.i] += self.ALIX[self.i]
      #self.DY[self.i] += self.ALIY[self.i]

  # BEGIN

  M = Meta()

  F = []
  for i in xrange(NUM):
    r = rand()*0.1
    alpha = rand()*pi*2.
    x = C + cos( alpha )*r
    y = C + sin( alpha )*r
    the = rand()*pi*2.
    dx = cos( the )*STP
    dy = sin( the )*STP
    dx = 0
    dy = 0
    F.append( Boid(x,y,dx,dy) )


  plt.figure(0)
  plt.ion()

  itt = 0
  ti = time()
  for itt in xrange(ITTMAX):

    if not itt % 10:
      plt.clf()
      plt.plot(M.X,M.Y,'ro')
      plt.axis([0,1,0,1])
      plt.draw()

    M.set_dist()
    M.set_flock()

    for boid in F:
      boid.separation()
      boid.alignment()
      boid.step()

    M.iterate()

      

if __name__ == '__main__' :
  main()

