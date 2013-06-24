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


  pii      = 2.*pi
  N        = 2000
  BACK     = 1.
  FRONT    = 0.
  ALPHA    = 0.5
  OUT      = 'b'
  C        = 0.5
  ITTMAX   = 10000

  ONE      = 1./N
  STP      = ONE*0.9

  NUM      = 200
  FR       = 0.01 # follow radius
  NR       = FR*0.9
  DR       = 10*FR    

  ## CLASSES

  class Meta(object):

    c   = [0]
    X   = zeros(NUM,dtype=np.float)
    Y   = zeros(NUM,dtype=np.float)
    dX  = zeros(NUM,dtype=np.float)
    dY  = zeros(NUM,dtype=np.float)
    THE = zeros(NUM,dtype=np.float)
    VEL = zeros(NUM,dtype=np.float)

    A   = zeros( (NUM,NUM), dtype=np.float )
    R   = zeros( (NUM,NUM), dtype=np.float )

    F   = zeros( (NUM,NUM), dtype=np.bool )


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
        inflock = d < FR
        inflock[i] = False
        self.F[i,:] = inflock[:]
        print inflock


  class Boid(Meta):

    def __init__(self, x,y,the,vel):
      self.i = self.c[0]
      self.c[0] += 1

      self.X[self.i] = x
      self.Y[self.i] = y
      self.THE[self.i] = the
      self.VEL[self.i] = vel

    def step(self):
      
      i = self.i
      self.X[i] += cos(self.THE[i])*self.VEL[i]
      self.Y[i] += sin(self.THE[i])*self.VEL[i]


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
      boid.step()

      #inflock    = d < FR
      #inflock[i] = False

      #if inflock.sum() > 1:
        #n = inflock.sum()
        #thex = cos(THE[inflock])
        #they = sin(THE[inflock])
        #dTHE[i] = arctan2(they.sum()/n,thex.sum()/n)

      #else:
        #dTHE[i] = THE[i]
      
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

