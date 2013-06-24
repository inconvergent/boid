#!/usr/bin/python
# -*- coding: utf-8 -*-

def main():

  import numpy as np
  #import scipy as sp
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
  ITTMAX = 10000

  N = 1000
  ONE = 1./N
  STP = ONE*0.9

  NUM = 200
  FLOCK_RAD = 0.01

  SEPARATION_PRI = 0.5
  ALIGNMENT_PRI = 0.3
  COHESION_PRI = 0.2

  ## CLASSES

  class Meta(object):

    c = [0]
    X = zeros(NUM,dtype=np.float)
    Y = zeros(NUM,dtype=np.float)
    THE = zeros(NUM,dtype=np.float)
    VEL = zeros(NUM,dtype=np.float)

    SEP_VEL = zeros(NUM,dtype=np.float)
    SEP_THE = zeros(NUM,dtype=np.float)
    ALI_VEL = zeros(NUM,dtype=np.float)
    ALI_THE = zeros(NUM,dtype=np.float)
    COH_VEL = zeros(NUM,dtype=np.float)
    COH_THE = zeros(NUM,dtype=np.float)

    A = zeros( (NUM,NUM), dtype=np.float )
    R = zeros( (NUM,NUM), dtype=np.float )

    F = zeros( (NUM,NUM), dtype=np.bool )

    def set_dist(self):

      self.R[:] = cdist(*[ colstack((self.X,self.Y)) ]*2 ) 

    def set_phi(self):

      for i in xrange(NUM):
        dx = self.X[i] - self.X
        dy = self.Y[i] - self.Y
        self.A[i,:] = arctan2(dy,dx)

    def set_flock(self):

      for i in xrange(NUM):
        d = self.R[i,:]
        inflock = d < FLOCK_RAD
        inflock[i] = False
        self.F[i,:] = inflock[:]

    def step(self):
      
      i = self.i
      self.
      self.X[i] += cos(self.THE[i])*self.VEL[i]
      self.Y[i] += sin(self.THE[i])*self.VEL[i]


  class Boid(Meta):

    def __init__(self, x,y,the,vel):
      self.i = self.c[0]
      self.c[0] += 1

      self.X[self.i] = x
      self.Y[self.i] = y
      self.THE[self.i] = the
      self.VEL[self.i] = vel

    def alignment(self):

      inflock = self.F[self.i,:]
      if inflock.sum() > 1:
        n = inflock.sum()
        thex = cos(self.THE[inflock])
        they = sin(self.THE[inflock])
        self.THE[self.i] = arctan2(they.sum()/n,thex.sum()/n)

      else:
        self.dTHE[self.i] = self.THE[self.i]

    #def step(self):
      
      #i = self.i
      #self.X[i] += cos(self.THE[i])*self.VEL[i]
      #self.Y[i] += sin(self.THE[i])*self.VEL[i]


  # BEGIN

  M = Meta()

  F = []
  for i in xrange(NUM):
    r = rand()*0.1
    the = rand()*pi*2.
    vel = STP
    x = C + cos(the)*r
    y = C + sin(the)*r
    F.append(Boid(x,y,the,vel))


  plt.figure(0)
  plt.ion()
    
  itt = 0
  ti = time()
  for itt in xrange(ITTMAX):

    M.set_dist()
    M.set_phi()
    M.set_flock()

    for boid in F:
      boid.alignment()

    M.step()



      
      ### find objects that are too close
      ### avoid colisions
      #toonear = d < NR
      #toonear[i] = False

      #if toonear.sum() > 1:
        ##speed = (NR - d[toonear])
        ##dX[toonear] -= cos(a[toonear])*speed
        ##dY[toonear] -= sin(a[toonear])*speed
        #dX[toonear] -= cos(a[toonear])
        #dY[toonear] -= sin(a[toonear])

      ### find objects that are too far off
      ### approach
      #toofar = xand(d < DR , d > FR)
      #toofar[i] = False

      #if toofar.sum() > 1:
        #dX[toofar] += cos(a[toofar])
        #dY[toofar] += sin(a[toofar])
      
    #rnd = (1.-2.*rand(NUM)) * 2.*pi/100.*2
    
    #THE[:] = dTHE[:] + rnd
    #dX += cos(THE)
    #dY += sin(THE)

    #Y += dY*STP
    #X += dX*STP

    ##THE += rnd

    #if not itt % 10:
      #plt.clf()
      #plt.plot(X,Y,'ro')
      #plt.axis([0,1,0,1])
      #plt.draw()
      

  ##sur.write_to_png('{:s}.png'.format(OUT))
  #print itt, time()-ti
  #ti = time()


if __name__ == '__main__' :
  main()

