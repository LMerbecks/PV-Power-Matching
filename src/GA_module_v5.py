# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:17:54 2024

@author: magnusj
Breyting frá v3 eru
- skil á bestu parametrum
"""

import random
import math
import numpy as np


# Genetic Algorithm ********************************************************
def rea_rand_population(Rlimit, nrvar, npop):
    # Set up an initial array of all zeros
    Rpop = np.zeros((npop, nrvar))
    Rmin = Rlimit[0][:]
    Rmax = Rlimit[1][:]
    # Loop through each row (individual)
    for i in range(npop):
        Rpop[i,:] = np.array([Rmin + (Rmax-Rmin)*np.random.rand(nrvar)])
    return Rpop

def int_rand_population(Ilimit, nivar, npop):
    # Set up an initial array of all zeros
    Ipop = np.zeros((npop, nivar)).astype(int)
    Imin = Ilimit[0][:]
    Imax = Ilimit[1][:]
    # Loop through each row (individual)
    for i in range(npop):
        Ipop[i,:] = np.array([Imin + np.around((Imax-Imin)*np.random.rand(nivar))])
    return Ipop

def per_rand_population(npvar, npop):
    # Set up an initial array of all zeros
    Ppop = np.zeros((npop,npvar)).astype(int)
    Ppop = np.argsort(np.random.random(size=(Ppop.shape)))
    return Ppop

def gen_parents(nnum):
    Ipar = np.zeros((2,nnum)).astype(int)
    Ipar = np.argsort(np.random.rand(2,nnum))
    return Ipar

def tournament_selection(Rpop,Ipop,Ppop,nrvar,nivar,npvar,PI,ptou):
    npop  = PI.shape[0]
    
    Rnew = np.zeros((npop,nrvar))
    Inew = np.zeros((npop,nivar)).astype(int)
    Pnew = np.zeros((npop,npvar)).astype(int)

    Ipar = gen_parents(npop)

    for i in range(npop):
        if(nrvar > 0): Rnew[i][:] = Rpop[Ipar[0][i]][:]
        if(nivar > 0): Inew[i][:] = Ipop[Ipar[0][i]][:]
        if(npvar > 0): Pnew[i][:] = Ppop[Ipar[0][i]][:]


        ptran = np.random.random(1)
        if ptran < ptou:
            if PI[Ipar[1][i]] < PI[Ipar[0][i]]: 
                if(nrvar > 0): Rnew[i][:] = Rpop[Ipar[1][i]][:]
                if(nivar > 0): Inew[i][:] = Ipop[Ipar[1][i]][:]
                if(npvar > 0): Pnew[i][:] = Ppop[Ipar[1][i]][:]
    return Rnew,Inew,Pnew

def rea_heuristic_crossover(Rold,Ipar,pxov):
    npop  = Rold.shape[0]
    nrvar = Rold.shape[1]
    Rpop = np.array(Rold)

    Rpar1 = np.zeros((nrvar))
    Rpar2 = np.zeros((nrvar))
    for i in range(npop):    
        pint = np.random.random(1)
        Rpar1 = Rold[Ipar[0][i]][:]
        Rpar2 = Rold[Ipar[1][i]][:]
        if pint < pxov:
            Rpop[i][:] = Rpar1[:]+np.random.rand(nrvar)*(Rpar2[:]-Rpar1[:])
        else:
            cross_1 = random.randint(0,nrvar)
            Rpop[i][0:cross_1]  = Rold[Ipar[0][i]][0:cross_1]
            if cross_1 > nrvar-1 : Rpop[i][cross_1:nrvar-1] = Rold[Ipar[1][i]][cross_1:nrvar-1]
    return Rpop

def rea_rand_mutation(Rold,Rlimit,pmut):
    npop  = Rold.shape[0]
    nrvar = Rold.shape[1]

    Rpop = np.array(Rold)
    Rmin = Rlimit[0][:]
    Rmax = Rlimit[1][:]
    for i in range(npop):
        prand = np.random.random(1)
        if prand <= pmut:
            Rpop[i][:] = Rmin[:]+np.random.random(nrvar)*(Rmax[:]-Rmin[:])
        # Return a mutated child
    return Rpop

def int_heuristic_crossover(Iold,Ipar,pxov):
    npop  = Iold.shape[0]
    nivar = Iold.shape[1]
    Ipop = np.array(Iold)

    Ipar1 = np.zeros((nivar)).astype(int)
    Ipar2 = np.zeros((nivar)).astype(int)
    for i in range(npop):    
        pint = np.random.random(1)
        Ipar1 = Iold[Ipar[0][i]][:]
        Ipar2 = Iold[Ipar[1][i]][:]
        if pint < pxov:
            Ipop[i][:] = Ipar1[:]+np.around(np.random.rand(nivar)*(Ipar2[:]-Ipar1[:]))
        else:
            cross_1 = random.randint(0,nivar)
            Ipop[i][0:cross_1]  = Iold[Ipar[0][i]][0:cross_1]
            if cross_1 > nivar-1 : Ipop[i][cross_1:nivar-1] = Iold[Ipar[1][i]][cross_1:nivar-1]
    return Ipop

def int_rand_mutation(Iold,Ilimit,pmut):
    npop  = Iold.shape[0]
    nivar = Iold.shape[1]

    Imut = np.array(Iold)
    Imin = Ilimit[0][:]
    Imax = Ilimit[1][:]
    for i in range(npop):
        prand = np.random.random(1)
        if prand <= pmut:
            Imut[i][:] = Imin[:]+np.floor(np.random.random(nivar)*(Imax[:]-Imin[:]))
#        Imut = Imin+np.around(np.random.random(nivar)*(Imax-Imin))
        # Return a mutated child
    return Imut

def per_edge_crossover(Pold,Ipar,pxov):
    npop  = Pold.shape[0]
    npvar = Pold.shape[1]
    Ppop = np.array(Pold)
    for i in range(npop):    
        pint = np.random.random(1)
        if pint < pxov:
            
             etab = np.zeros((npvar,4)).astype(int)
            
             xpar1 = Pold[Ipar[0][i]][:]
             xpar2 = Pold[Ipar[1][i]][:]
            
             for j in range(npvar):
                 iteind = np.where(xpar1==j)[0][0]
                 if(iteind==0):
                     etab[j][0]=xpar1[npvar-1]
                     etab[j][1]=xpar1[1]
                 elif(iteind==npvar-1):
                     etab[j][0]=xpar1[npvar-2]
                     etab[j][1]=xpar1[0]
                 else:
                     etab[j][0]=xpar1[iteind-1]
                     etab[j][1]=xpar1[iteind+1]
            
             for j in range(npvar):
                 iteind = np.where(xpar2==j)[0][0]
                 if(iteind==0):
                     etab[j][2]=xpar2[npvar-1]
                     etab[j][3]=xpar2[1]
                 elif(iteind==npvar-1):
                     etab[j][2]=xpar2[npvar-2]
                     etab[j][3]=xpar2[0]
                 else:
                     etab[j][2]=xpar2[iteind-1]
                     etab[j][3]=xpar2[iteind+1]
                 etab[j][:] = np.sort(etab[j][:])
                 etab[j][0:len(np.unique(etab[j][:]))]=np.unique(etab[j][:])
                 etab[j][len(np.unique(etab[j][:])):4]= -1
                     
             icity = math.floor(npvar*np.random.random(1))
             xoff = np.zeros(npvar).astype(int)
             xoff[0] = icity
             for j in range(1,npvar):
                 etab[np.where(etab==icity)] = -1
                 inex = etab[icity][np.where(etab[icity][:]!=-1)]
                 inex = inex.astype(int)
                 if(len(inex) > 0):
                     nnex = np.ones(4)*5
                     for k in range(len(inex)):
                         nnex[k] = len(np.where(etab[inex[k]][:]!=-1)[0][:])
                     iran = np.where(nnex == min(nnex))
                     icity = inex[iran[0][math.floor(len(iran)*np.random.random(1))]]
                 else:
                     nnex = etab[np.where(etab[:][:]!=-1)]
                     icity = nnex[math.floor(len(nnex)*np.random.random(1))].astype(int)
                 xoff[j] = icity
             Ppop[i][:] = xoff[:]
    # Return children
    return Ppop

def per_order_crossover(Pold,Ipar,pxov):
    npop  = Pold.shape[0]
    npvar = Pold.shape[1]
    Ppop = np.array(Pold)
    Ppar1 = np.array(npvar).astype(int)
    Ppar2 = np.array(npvar).astype(int)

    for i in range(npop):    
        pint = np.random.random(1)
        if pint < pxov:
            # Pick crossover point, avoding ends of chromsome
            cross_1 = random.randint(0,npvar-2)
            cross_2 = random.randint(cross_1+1,npvar)
            # Parent first step
            Ppar1 = Pold[Ipar[0][i]][cross_1:cross_2]
            Ppar2= Pold[Ipar[1][i]][:]
            # Parent next step
            Pres =  [ele for ele in Ppar2 if ele not in Ppar1]
            Ppar1 = np.int_( np.append(Ppar1, Pres))
            # Parent reordering the string
            Poff = Ppar1[cross_2-cross_1:cross_2] 
            Poff = np.append(Poff, Ppar1[0:cross_2-cross_1])  
            Poff = np.append(Poff, Ppar1[cross_2:npvar])  
            Ppop[i][:] = Poff[:]
    # Return children
    return Ppop


def per_rand_mutation(Pold,pmut):
    npop  = Pold.shape[0]
    npvar = Pold.shape[1]
    Ppop = np.array(Pold)
    # Pick crossover point, avoding ends of chromsome
    prand = np.random.random(1)
    for i in range(npop):
        if prand <= pmut:
            imut_1 = random.randint(0,npvar-1)
            imut_2 = random.randint(0,npvar-1)
            Ppop[i][imut_1] = Pold[i][imut_2]
            Ppop[i][imut_2] = Pold[i][imut_1]
    # Return a mutated child
    return Ppop


#*****************************************************************************
# Main Algorithm code
# Now we'll go through the generations of genetic algorithm
# Útgáfa 29.08.24 kl. 13:12


def ga_min(obj_file,obj_fun,nvar,Ilimit,Rlimit,pvar):
    loaded_module = __import__(obj_file)
    PI_function = getattr(loaded_module,obj_fun)


    ngen = nvar[0]
    npop = nvar[1]
    nrvar = nvar[2]  # Number of real decision variables
    nivar = nvar[3]  # Number of integer decision variables
    npvar = nvar[4]  # Number of permutation decision variables

    ptou = pvar[0]   # Probability of tournament selection
    pxov = pvar[1]   # Probability of crossover
    pmut = pvar[2]   # Probability of mutation


    PI_glo = 0.0
    PI_ave = 0.0
    PI_max = 0.0

    nga = 1

    for iga in range(nga):
        PI_best_progress = [] # Tracks progress
        
        if(nrvar > 0): Rpop = rea_rand_population(Rlimit,nrvar,npop)
        else: Rpop = np.zeros((npop,nrvar))
        if(nivar > 0): Ipop = int_rand_population(Ilimit,nivar,npop)
        else: Ipop = np.zeros((npop,nivar)).astype(int)
        if(npvar > 0): Ppop = per_rand_population(npvar,npop)
        else: Ppop = np.zeros((npop,npvar)).astype(int)
        

        if(nrvar > 0): Rbest = np.array([Rpop[0][:]])
        else: Rbest = np.zeros(0)
        if(nivar > 0): Ibest = np.array([Ipop[0][:]])
        else: Ibest = np.zeros(0).astype(int)
        if(npvar > 0): Pbest = np.array([Ppop[0][:]])
        else: Pbest = np.zeros(0).astype(int)

#        PI =  fitness_function(Rpop,Ipop,Ppop)
        PI = np.zeros((npop))
        for ipop in range(npop):
            PI[ipop] = PI_function(Rpop[ipop][:],Ipop[ipop][:],Ppop[ipop][:])


        PI_best = np.min(PI)
        ind = np.where(PI == PI_best)
        if(nrvar > 0): Rbest = np.array(Rpop[ind[0]][:])
        if(nivar > 0): Ibest = np.array(Ipop[ind[0]][:])
        if(npvar > 0): Pbest = np.array(Ppop[ind[0]][:])

        print ('Starting best score, %0.3f: ' % PI_best)
        # Add starting best score to progress tracker
        PI_best_progress.append(PI_best)
        
        
        for igen in range(ngen):
        

            Rpop,Ipop,Ppop = tournament_selection(Rpop,Ipop,Ppop,nrvar,nivar,npvar,PI,ptou)
        
            Ipar = gen_parents(npop)
            if(nrvar > 0):
                Rpop = rea_heuristic_crossover(Rpop,Ipar,pxov)
                Rpop = rea_rand_mutation(Rpop,Rlimit,pmut)

            if(nivar > 0):
                Ipop = int_heuristic_crossover(Ipop,Ipar,pxov)
                Ipop = int_rand_mutation(Ipop,Ilimit,pmut)
            
            if(npvar > 0):
                Ppop = per_order_crossover(Ppop,Ipar,pxov)
#                Ppop = per_edge_crossover(Ppop,Ipar,pxov)
                Ppop = per_rand_mutation(Ppop,pmut)
        


#            PI =  fitness_function(Rpop,Ipop,Ppop)
            PI = np.zeros((npop))
            for ipop in range(npop):
                PI[ipop] = PI_function(Rpop[ipop][:],Ipop[ipop][:],Ppop[ipop][:])
        

            # Add starting best score to progress tracker
            PI_best_progress.append(PI_best)
        
            if np.min(PI) < PI_best:
                PI_best = np.min(PI)
#                ind = np.zeros(1)
                ind = np.where(PI == PI_best)
#                print(ind,PI[ind])
                if(nrvar > 0): Rbest = np.array(Rpop[ind[0][0]][:])
                if(nivar > 0): Ibest = np.array(Ipop[ind[0][0]][:])
                if(npvar > 0): Pbest = np.array(Ppop[ind[0][0]][:])

#                print('Number of runs:   ',iga)
                print('Number of generations: ', igen)    
                print ('Best score: %0.3f' % (PI_best))
                print('Best individual: ')
                if(nrvar > 0): print("Real variable:       ", Rbest[:])
                if(nivar > 0): print("Integer variable:    ", Ibest[:])
                if(npvar > 0): print("Permutation variable:",Pbest[:])
                print()
                ico = 0
                PI[0] = PI_best
                if(nrvar > 0): Rpop[0,:] = Rbest[:]
                if(nivar > 0): Ipop[0,:] = Ibest[:]
                if(npvar > 0): Ppop[0,:] = Pbest[:]

        if(iga == 0): PI_glo = PI_best
        if(PI_best < PI_glo): PI_glo = PI_best
        PI_ave = PI_ave + PI_best
        if(PI_best > PI_max): PI_max = PI_best
#        print(iga, PI_glo, PI_ave/(iga+1), PI_max)

    return PI_best,Rbest,Ibest,Pbest,PI_best_progress
# GA has completed required generation


