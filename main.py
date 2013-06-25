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

  BLACK_HOLE = STP

  FUZZ = STP
  #FUZZ = 0.

  NUM = 50
  FLOCK_RAD = 0.1
  NEAR_RAD = FLOCK_RAD*0.2
  COHESION_RAD = FLOCK_RAD

  SEPARATION_PRI = 0.5
  ALIGNMENT_PRI = 0.4
  COHESION_PRI = 0.1

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

    def iterate(self):

      alpha = rand( NUM ) *pi*2.
      self.DX[:] += cos( alpha )*FUZZ
      self.DY[:] += sin( alpha )*FUZZ

      self.X[:] += self.DX[:]
      self.Y[:] += self.DY[:]

      self.DX[:] += self.SEPX[:]*SEPARATION_PRI + \
                        self.ALIX[:]*ALIGNMENT_PRI
      self.DY[:] += self.SEPY[:]*SEPARATION_PRI + \
                        self.ALIY[:]*ALIGNMENT_PRI

      self.SEPX[:] = 0.
      self.SEPY[:] = 0.
      self.ALIX[:] = 0.
      self.ALIY[:] = 0.

  class Boid(Meta):

    def __init__(self, x,y,dx,dy):
      self.i = self.c[0]
      self.c[0] += 1

      self.X[self.i] = x
      self.Y[self.i] = y
      self.DX[self.i] = dy
      self.DY[self.i] = dx

    def separation(self):

      d = self.R[self.i,:]
      inflock = d < NEAR_RAD
      inflock[self.i] = False

      if inflock.sum() > 0:

        scale = 1./square( self.R[self.i,inflock] )

        dx = ( self.X[self.i] - self.X[inflock] ) * scale
        dy = ( self.Y[self.i] - self.Y[inflock] ) * scale
 
        sx = dx.sum() / N
        sy = dy.sum() / N

        tot = sqrt(square(sx) + square(sy))
        if tot>BLACK_HOLE:
          sx = BLACK_HOLE*sx/tot
          sy = BLACK_HOLE*sy/tot

        self.SEPX[self.i] = sx
        self.SEPY[self.i] = sy

    def alignment(self):

      d = self.R[self.i,:]
      inflock = d < FLOCK_RAD
      inflock[self.i] = False

      if inflock.sum() > 0:

        scale = 1./square( self.R[self.i,inflock] )

        dx = ( self.DX[self.i] - self.DX[inflock] ) * scale
        dy = ( self.DY[self.i] - self.DY[inflock] ) * scale

        sx = dx.sum() / N
        sy = dy.sum() / N

        tot = sqrt(square(sx) + square(sy))
        if tot>BLACK_HOLE:
          sx = BLACK_HOLE*sx/tot
          sy = BLACK_HOLE*sy/tot

        self.ALIX[self.i] = -sx
        self.ALIY[self.i] = -sy

    def cohesion(self):

      d = self.R[self.i,:]
      inflock = d < COHESION_RAD
      inflock[self.i] = False

      if inflock.sum() > 0:

        scale = 1./square( self.R[self.i,inflock] )

        dx = ( self.X[self.i] - self.X[inflock] ) * scale
        dy = ( self.Y[self.i] - self.Y[inflock] ) * scale

        sx = dx.sum() / N
        sy = dy.sum() / N

        tot = sqrt(square(sx) + square(sy))
        if tot>BLACK_HOLE:
          sx = BLACK_HOLE*sx/tot
          sy = BLACK_HOLE*sy/tot

        self.SEPX[self.i] = -sx
        self.SEPY[self.i] = -sy


  # BEGIN

  M = Meta()

  F = []
  for i in xrange(NUM):
    r = rand()*0.3
    alpha = rand()*pi
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

    for boid in F:
      boid.separation()
      boid.alignment()
      boid.cohesion()

    M.iterate()

      

if __name__ == '__main__' :
  main()

