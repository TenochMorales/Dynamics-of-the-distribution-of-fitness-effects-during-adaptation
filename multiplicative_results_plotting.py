# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 05:48:36 2024

@author: tenoc
"""

import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
from numba.core.decorators import njit
import time




@njit
def computeFitnessFunc(w,Ldist):
    index = np.argmin(np.abs(wVector - w))
    return w*Ldist[index]
    
def computeFitness(Ldist,lowest_fitness_contribution,highest_fitness_contribution):
    
    result = sp.quad(computeFitnessFunc, lowest_fitness_contribution, highest_fitness_contribution, args=(Ldist),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    return result



# Gives integrand in equation 6 from overleaf (and so also equation 5)
@njit
def LFixFunc(wPrime, w, Ldist, Mdist,Fitness):
    index = np.argmin(np.abs(wVector - wPrime))
    fVal = 2 * ((wPrime**(1/N) - w**(1/N)) / w**(1/N))
    return Mdist[index] * fVal



# Gives integrand in equation 7 from overleaf
@njit
def MFixFunc(w,wPrime,Ldist,Mdist,Fitness):
    index = np.argmin(np.abs(wVector - w))
    fVal = 2 *((wPrime**(1/N) - w**(1/N)) / w**(1/N))

    return Ldist[index] * fVal



def fixProbFunc(w, Ldist, Mdist,Fitness,lowest_fitness_contribution,highest_fitness_contribution):
    index = np.argmin(np.abs(wVector-w))
    integrand = sp.quad(MFixFunc, lowest_fitness_contribution, w, args=(w,Ldist,Mdist,Fitness),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0] # w, 1
    return Mdist[index]*integrand



# Gives equation 5 in overleaf
def computeFixProb(Ldist, Mdist,Fitness,lowest_fitness_contribution,highest_fitness_contribution):
    result = sp.quad(fixProbFunc,lowest_fitness_contribution, highest_fitness_contribution, args=(Ldist,Mdist,Fitness,lowest_fitness_contribution,highest_fitness_contribution),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0] # 0, 1
    return result



# Gives equation 6 in overleaf
def makeLFdist(Ldist, Mdist,FProb, Fitness,lowest_fitness_contribution,highest_fitness_contribution):
    LFdist = np.zeros(len(Ldist))
    for i in range(len(LFdist)):
        LFdist[i] = (Ldist[i]/FProb)*sp.quad(LFixFunc, wVector[i], highest_fitness_contribution, args=(wVector[i], Ldist, Mdist, Fitness),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04,  )[0] # wVector[i], 1

    return LFdist


# Gives equation 7 in overleaf
def makeMFdist(Ldist, Mdist, FProb,Fitness,lowest_fitness_contribution,highest_fitness_contribution):

    MFdist = np.zeros(len(Mdist))
    for i in range(len(Mdist)):
        MFdist[i] = (Mdist[i]/FProb)*sp.quad(MFixFunc, lowest_fitness_contribution, wVector[i], args=(wVector[i], Ldist, Mdist,Fitness),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04,  )[0]

    return MFdist


# Gives equation 9 in overleaf
@njit
def nextL(Ldist,LFdist,MFdist):
    nextLdist = np.zeros(len(Ldist))
    for i in range(len(Ldist)):
        nextLdist[i] = Ldist[i] + MFdist[i]/N - LFdist[i]/N

    return nextLdist


# Gives equation 10 in overleaf
@njit
def nextM(Mdist, LFdist, MFdist,B,numWs):
    nextMdist = np.zeros(len(Mdist))
    for i in range(len(Mdist)):
        nextMdist[i] = Mdist[i] + LFdist[i]/((B-1)*N) - MFdist[i]/((B-1)*N)

    return nextMdist



############################################################
def computeDists(m,B,numWs):
    
    lowest_fitness_contribution = 0.5
    highest_fitness_contribution = 1.5
    
    
    allLdists = np.zeros((m + 1, numWs))
    allMdists = np.zeros((m + 1, numWs))
    allLFdists = np.zeros((m + 1, numWs))
    allMFdists = np.zeros((m + 1, numWs))
    
    initialLdist = np.zeros(numWs)
    initialMdist = np.zeros(numWs)

    ## DIFFERENT INITAL DISTS (comment out to just use uniform) ##
    i = 0
    wVector = np.linspace(lowest_fitness_contribution,highest_fitness_contribution, num = numWs)
    for w in wVector:
        initialLdist[i] = 1#2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        initialMdist[i] = 1#2-2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        i = i + 1
        
    Ldist = initialLdist
    Mdist = initialMdist


    for i in range(m + 1):

        Fitness = computeFitness(Ldist,lowest_fitness_contribution,highest_fitness_contribution)
        FProb = computeFixProb(Ldist, Mdist,Fitness,lowest_fitness_contribution,highest_fitness_contribution)
        print("Mutation =",i,"of",str(m))

        allLdists[i] = Ldist
        allMdists[i] = Mdist

        LFdist = makeLFdist(Ldist, Mdist, FProb, Fitness,lowest_fitness_contribution,highest_fitness_contribution)
        MFdist = makeMFdist(Ldist, Mdist, FProb, Fitness,lowest_fitness_contribution,highest_fitness_contribution)

        allLFdists[i] = LFdist
        allMFdists[i] = MFdist

        Ldist = nextL(Ldist, LFdist, MFdist)
        Mdist = nextM(Mdist, LFdist, MFdist,B,numWs)



    return allLdists, allLFdists, allMdists, allMFdists



def expectation(w,dist,wVector):
    index = np.argmin(np.abs(np.array(wVector) - w))
    pdf = dist[index]
    return w*pdf


def variance(w,dist,mean,wVector):
    
    index = np.argmin(np.abs(np.array(wVector) - w))
    pdf = dist[index]
    return (w-mean)**2*pdf


def fitness_over_time(Ldist_vector,Mdist_vector):
    time_vector = np.zeros(len(Ldist_vector))
    fitness_vector = np.zeros(len(Ldist_vector))
    fitness_vector[0] = computeFitness(Ldist_vector[0],lowest_fitness_contribution,highest_fitness_contribution)
    for i in range(1,len(Ldist_vector)):
        fitness_vector[i] = computeFitness(Ldist_vector[i],lowest_fitness_contribution,highest_fitness_contribution)
        time_vector[i] = time_vector[i-1] + 1/computeFixProb(Ldist_vector[i],Mdist_vector[i],lowest_fitness_contribution,highest_fitness_contribution)
    
    return np.array([time_vector,fitness_vector])


@njit
def check_normalizationFunc(w,distribution):
    
    index = np.argmin(np.abs(np.array(wVector) - w))
    pdf = distribution[index]
    return pdf

def check_normalization(distribution):
    
    result = sp.quad(check_normalizationFunc, .5, 1.5, args=(distribution),limit = 10**4,epsabs=1.0e-04, epsrel=1.0e-04,  )[0]
    return result





def integrand_dfe(w,s,Ldist,Mdist,wVector,W):
    
    Lindex = np.argmin(np.abs(np.array(wVector)-w))
    L = Ldist[Lindex]

    wPrime = w*(1+s)**N
    Mindex = np.argmin(np.abs(np.array(wVector)-wPrime))
    M = Mdist[Mindex]

    return L*M*w*N*(1+s)**(N-1)


def computeDFE(Ldist,Mdist,wVector,lowest_fitness_contribution,highest_fitness_contribution):
    
    
    W = sp.quad(expectation,lowest_fitness_contribution,highest_fitness_contribution,args=(Ldist,wVector),limit = 10**4,epsabs=1.0e-04, epsrel=1.0e-04,  )[0]
    
    lowest_s = (lowest_fitness_contribution/highest_fitness_contribution)**(1/N) - 1
    highest_s = (highest_fitness_contribution/lowest_fitness_contribution)**(1/N) - 1
    sDist = np.linspace(lowest_s,highest_s, num =len(wVector)) # sDist = np.linspace(-1/(N*W), 1/(N*W), num = 1000)
    DFE = np.zeros(len(sDist))
    i = 0
    for s in sDist:
        if s <0:
            lowBound = lowest_fitness_contribution/((1+s)**N)
            val = sp.quad(integrand_dfe,lowBound,highest_fitness_contribution,args = (s,Ldist,Mdist,wVector,W),limit = 10**4,epsabs=1.0e-03, epsrel=1.0e-03) # lowBound, 1
        elif s > 0:
            uppBound = highest_fitness_contribution/((1+s)**N)
            val = sp.quad(integrand_dfe,lowest_fitness_contribution,uppBound,args = (s,Ldist,Mdist,wVector,W),limit = 10**4,epsabs=1.0e-03, epsrel=1.0e-03) # 0, uppBound
    
        DFE[i]+=val[0]
        i+=1
        
    return sDist, DFE


#############
#############
############# Simulations for multiplicative landscape
#############
#############
import numpy.random as rnd
from numba.core.decorators import njit







@njit
def set_fitness_contributions(numberLoci, initialDistribution,B):

    fitnessContributions = np.zeros((numberLoci,B))
    for i in range(numberLoci):
        for j in range(B):
            fitnessContributions[i][j] = rnd.rand()+0.5

    return fitnessContributions
############################################################################



## .......... SELECT A MUTATION WITH A CERTAIN BIAS .......... ##
@njit
def selectMutation(genome,B):

    mutatedGenome = genome.copy()
    mutatedLoci = rnd.randint(0,len(genome))
    mutation = rnd.randint(0,B-1) + 1
    mutatedGenome[mutatedLoci] = np.mod(genome[mutatedLoci] + mutation, B)
    
    return mutatedGenome, mutatedLoci







## .......... COMPUTE FITNESS .......... ##
@njit
def compute_fitness(genome, fitnessContributions):
    
    Fitness = 1
    for i in range(len(genome)):
        Fitness *= fitnessContributions[i][genome[i]]

    return Fitness**(1/len(genome))








## .......... COMPUTE DISTRIBUTIONS L,Q,M,S .......... ##
@njit
def computeDistributions(genome,fitnessContributions,B):
    
    distributionL = np.zeros(len(genome))
    distributionM = np.zeros((B-1)*len(genome))
    for i in range(len(genome)):
        distributionL[i] = fitnessContributions[i][genome[i]]
        for a in range(B-1):
            mutationIndex = np.mod(genome[i] + a + 1, B)
            distributionM[(B-1)*i + a] = fitnessContributions[i][mutationIndex]

    return distributionL, distributionM





## ADJUSTED DFE: ##
@njit
def compute_fitness_effects_distribution(initialGenome, fitnessContributions,B):

    DFE = np.zeros((B-1)*len(initialGenome))

    for i in range(len(initialGenome)):
        for j in range(B-1):
            #DFE[(B-1)*i+j] = (fitnessContributions[i][np.mod(initialGenome[i] + j+1,B)]**(1/N) - fitnessContributions[i][initialGenome[i]]**(1/N))/(fitnessContributions[i][initialGenome[i]]**(1/N))
            mutation_genome = initialGenome.copy()
            mutation_genome[i] = np.mod(initialGenome[i] + j+1,B)
            DFE[(B-1)*i+j] = compute_fitness(mutation_genome, fitnessContributions)/compute_fitness(initialGenome, fitnessContributions) - 1
    return DFE



# .......... SELECTION AND OUTCOME OF MUTATION .......... ##
@njit
def mutationTryAndOutcome(genome,fitnessContributions,B):

    mutatedGenome, mutatedLoci = selectMutation(genome,B)

    s = -1 + compute_fitness(mutatedGenome,fitnessContributions)/compute_fitness(genome,fitnessContributions)
    success = False
    if s > 0:
        if 2*s > rnd.rand():
            success = True
    return success, mutatedGenome





## .......... ADAPTIVE WALK .......... ##
@njit
def compute_addaptive_step(genome, fitnessContributions,B):
    
    mutations = 0 
    tries = 0
        
    while mutations == 0:
        success, mutatedGenome = mutationTryAndOutcome(genome, fitnessContributions,B)
        if success :  ## if a mutation happened
            genome = mutatedGenome.copy()
            mutations += 1
        
        tries += 1
        if tries > 10**5:
            print('Maximum tries surpassed')
            mutations += 1
            genome = mutatedGenome.copy()
    return genome, tries




@njit
def samplingInWalks(N,B,numberWalks,numberMutations):
    
    walk = 0
    samples_loci_distribution = np.zeros( (numberMutations+1,numberWalks*N) )
    samples_mutations_distribution = np.zeros( (numberMutations+1,numberWalks*N*(B-1)) )
    samples_fixed_loci_distribution = np.zeros( (numberMutations+1,numberWalks) )
    samples_fixed_mutations_distribution = np.zeros( (numberMutations+1,numberWalks) )
    mutations_until_fixation = np.zeros( (numberMutations+1,numberWalks) )

    samples_fitness_effects_distribution = np.zeros((numberMutations+1,numberWalks*N*(B-1)))

    while walk < numberWalks:
        
        
        
        if (walk*10/numberWalks) - int(walk*10/numberWalks) == 0:
            print("Walk: ",walk)
            
            
        initialGenome = np.zeros(N,dtype=np.int32)
        fitnessContributions = set_fitness_contributions(N,0,B)
        finalGenome = initialGenome.copy()
        for mutation in range(numberMutations + 1):
            
            beforeMutGenome = finalGenome.copy()
            finalGenome, tries = compute_addaptive_step(beforeMutGenome, fitnessContributions,B)
            mutations_until_fixation[mutation][walk] = tries 
            L, M = computeDistributions(beforeMutGenome, fitnessContributions,B)
            for i in range(N):
                samples_loci_distribution[mutation][walk*N + i] = L[i]
                for a in range(B-1):
                    samples_mutations_distribution[mutation][(walk*N*(B-1) + i*(B-1) + a)] = M[i*(B-1) + a]
                    
                if beforeMutGenome[i] != finalGenome[i]:
                    samples_fixed_loci_distribution[mutation][walk] = fitnessContributions[i][beforeMutGenome[i]]
                    samples_fixed_mutations_distribution[mutation][walk] = fitnessContributions[i][finalGenome[i]]

            DFE = compute_fitness_effects_distribution(beforeMutGenome,fitnessContributions,B)

            for i in range(len(DFE)):
                samples_fitness_effects_distribution[mutation][(walk*N*(B-1) + i)] = DFE[i]

        walk += 1

    return samples_loci_distribution, samples_mutations_distribution, samples_fixed_loci_distribution, samples_fixed_mutations_distribution, samples_fitness_effects_distribution, mutations_until_fixation




######
####
#####
######...............INFINITE ALLELES EXTRA FUNCTIONS .................#####
#####
####
######









# Gives equation 10 in overleaf
@njit
def nextM_infAlleles(Mdist, LFdist, MFdist,B,numWs):
    nextM_infAllelesdist = np.zeros(len(Mdist))
    for i in range(len(Mdist)):
        nextM_infAllelesdist[i] = Mdist[i] #+ LFdist[i]/((B-1)*N) - MFdist[i]/((B-1)*N)

    return nextM_infAllelesdist



############################################################
def computeDists_infAlleles(m,B,numWs):

        
    
    lowest_fitness_contribution = 0.5
    highest_fitness_contribution = 1.5
    
    allLdists = np.zeros((m + 1, numWs))
    allMdists = np.zeros((m + 1, numWs))
    allLFdists = np.zeros((m + 1, numWs))
    allMFdists = np.zeros((m + 1, numWs))
    
    initialLdist = np.zeros(numWs)
    initialMdist = np.zeros(numWs)

    ## DIFFERENT INITAL DISTS (comment out to just use uniform) ##
    i = 0
    wVector = np.linspace(lowest_fitness_contribution, highest_fitness_contribution, num = numWs)
    for w in wVector:
        initialLdist[i] = 1#2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        initialMdist[i] = 1#2-2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        i = i + 1
        
    Ldist = initialLdist
    Mdist = initialMdist


    for i in range(m + 1):

        Fitness = computeFitness(Ldist,lowest_fitness_contribution,highest_fitness_contribution)
        FProb = computeFixProb(Ldist, Mdist,Fitness,lowest_fitness_contribution,highest_fitness_contribution)
        print("Mutation =",i,"of",str(m))

        allLdists[i] = Ldist
        allMdists[i] = Mdist

        LFdist = makeLFdist(Ldist, Mdist, FProb, Fitness,lowest_fitness_contribution,highest_fitness_contribution)
        MFdist = makeMFdist(Ldist, Mdist, FProb, Fitness,lowest_fitness_contribution,highest_fitness_contribution)

        allLFdists[i] = LFdist
        allMFdists[i] = MFdist

        Ldist = nextL(Ldist, LFdist, MFdist)
        Mdist = nextM_infAlleles(Mdist, LFdist, MFdist,B,numWs)



    return allLdists, allLFdists, allMdists, allMFdists




# .......... SELECTION AND OUTCOME OF MUTATION .......... ##
@njit
def mutationTryAndOutcome_inf_alleles(genome,fitnessContributions,B):

    mutatedGenome, mutatedLoci = selectMutation(genome,B)
    fitnessContributions[mutatedLoci][mutatedGenome[mutatedLoci]] = rnd.rand() + 0.5
    s = -1 + compute_fitness(mutatedGenome,fitnessContributions)/compute_fitness(genome,fitnessContributions)
    success = False
    if s > 0:
        if 2*s > rnd.rand():
            success = True
    return success, mutatedGenome, fitnessContributions






## .......... ADAPTIVE WALK .......... ##
@njit
def compute_addaptive_step_inf_alleles(genome, fitnessContributions,B):
    
    mutations = 0 
    tries = 0
        
    while mutations == 0:
        success, mutatedGenome, newFitnessContributions = mutationTryAndOutcome_inf_alleles(genome, fitnessContributions,B)
        if success :  ## if a mutation happened
            genome = mutatedGenome.copy()
            mutations += 1
        
        tries += 1
        if tries > 10**7:
            print('Maximum tries surpassed')
            mutations += 1
    
    return genome, tries, newFitnessContributions




@njit
def samplingInWalksInfAlleles(N,B,numberWalks,numberMutations):
    
    walk = 0
    samples_loci_distribution = np.zeros( (numberMutations+1,numberWalks*N) )
    samples_mutations_distribution = np.zeros( (numberMutations+1,numberWalks*N*(B-1)) )
    samples_fixed_loci_distribution = np.zeros( (numberMutations+1,numberWalks) )
    samples_fixed_mutations_distribution = np.zeros( (numberMutations+1,numberWalks) )
    mutations_until_fixation = np.zeros( (numberMutations+1,numberWalks) )

    samples_fitness_effects_distribution = np.zeros((numberMutations+1,numberWalks*N*(B-1)))

    while walk < numberWalks:
        
        
        
        if (walk*10/numberWalks) - int(walk*10/numberWalks) == 0:
            print("Walk: ",walk)
            
            
        initialGenome = np.zeros(N,dtype=np.int32)
        fitnessContributions = set_fitness_contributions(N,0,B)
        finalGenome = initialGenome.copy()
        for mutation in range(numberMutations + 1):
            
            beforeMutGenome = finalGenome.copy()
            finalGenome, tries, fitnessContributions = compute_addaptive_step_inf_alleles(beforeMutGenome, fitnessContributions,B)
            mutations_until_fixation[mutation][walk] = tries 
            L, M = computeDistributions(beforeMutGenome, fitnessContributions,B)
            for i in range(N):
                samples_loci_distribution[mutation][walk*N + i] = L[i]
                for a in range(B-1):
                    samples_mutations_distribution[mutation][(walk*N*(B-1) + i*(B-1) + a)] = M[i*(B-1) + a]
                    
                if beforeMutGenome[i] != finalGenome[i]:
                    samples_fixed_loci_distribution[mutation][walk] = fitnessContributions[i][beforeMutGenome[i]]
                    samples_fixed_mutations_distribution[mutation][walk] = fitnessContributions[i][finalGenome[i]]

            DFE = compute_fitness_effects_distribution(beforeMutGenome,fitnessContributions,B)

            for i in range(len(DFE)):
                samples_fitness_effects_distribution[mutation][(walk*N*(B-1) + i)] = DFE[i]

        walk += 1

    return samples_loci_distribution, samples_mutations_distribution, samples_fixed_loci_distribution, samples_fixed_mutations_distribution, samples_fitness_effects_distribution, mutations_until_fixation






overall_time_start = time.time()

















numWs = 10**4
lowest_fitness_contribution = 0.5
highest_fitness_contribution = 1.5
wVector = np.linspace(lowest_fitness_contribution, highest_fitness_contribution, num = numWs)

N = 100
B = 20
m = 85
mutation_points20 = np.linspace(0,m,int(m/5)+1)

## GENERATES THE ANALYTICAL PREDICTIONS.....................................................................................................................................................................
allLdists20, allLFdists20, allMdists20, allMFdists20 = computeDists(m, B, numWs)
fitness_effects_values20, DFE20 = computeDFE( allLdists20[-1], allMdists20[-1], wVector,lowest_fitness_contribution,highest_fitness_contribution)

mean_L_analytical20 = np.zeros(len(mutation_points20))
mean_M_analytical20 = np.zeros(len(mutation_points20))
mean_LF_analytical20 = np.zeros(len(mutation_points20))
mean_MF_analytical20 = np.zeros(len(mutation_points20))
for i in range(len(mutation_points20)):
    mean_L_analytical20[i] = sp.quad(expectation,0.5,1.5,args=(allLdists20[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_M_analytical20[i] = sp.quad(expectation,0.5,1.5,args=(allMdists20[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_LF_analytical20[i] = sp.quad(expectation,0.5,1.5,args=(allLFdists20[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_MF_analytical20[i] = sp.quad(expectation,0.5,1.5,args=(allMFdists20[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    
analytical_fixation_time20 = np.zeros(m+1)
analytical_mean_fitness20 = np.zeros(m+1)
for i in range(m+1):
    analytical_fixation_time20[i] = 1/computeFixProb(allLdists20[i],allMdists20[i],computeFitness(allLdists20[i],lowest_fitness_contribution,highest_fitness_contribution),lowest_fitness_contribution,highest_fitness_contribution)
    analytical_mean_fitness20[i] = sp.quad(expectation,0.5,1.5,args=(allLdists20[int(i)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    





## GENERATES PLOTS WITH ANALYTICAL LINES AND SIMULATED HISTOGRAMS...........................................................................................................................................
fig, axs = plt.subplots(2,2,sharex=True,dpi=150, layout='constrained')
axs[1,1].plot([],[],color=(0.7,0.7,0.7))
axs[1,1].plot([],[],color=(0.3,0.3,0.3))
axs[0,0].set_ylabel("Probability density, $L_i$")
axs[0,1].set_ylabel("Probability density, $M_i$")
axs[1,0].set_xlabel('Fitness contribution, $w$')
axs[1,0].set_ylabel("Probability density, $L_{i|F}$")
axs[1,1].set_xlabel("Fitness contribution, $w'$ ")
axs[1,1].set_ylabel("Probability density, $M_{i|F}$")
axs[1,1].legend(('i = 0','i = 85'))

axs[0,0].plot(wVector,allLdists20[0],color=(0,0,1))
axs[0,0].plot(wVector,allLdists20[-1],color=(0,0,0.5))
axs[0,1].plot(wVector,allMdists20[0],color=(1,0,0))
axs[0,1].plot(wVector,allMdists20[-1],color=(0.5,0,0))
axs[1,0].plot(wVector,allLFdists20[0],color=(0,1,1))
axs[1,0].plot(wVector,allLFdists20[-1],color=(0,0.4,0.4))
axs[1,1].plot(wVector,allMFdists20[0],color=(1,0,1))
axs[1,1].plot(wVector,allMFdists20[-1],color=(0.5,0,0.5))

del allLFdists20
del allMFdists20

dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B20_initial.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B20_final.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,0.5))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B20_initial.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B20_final.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_B20_initial.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,1,1))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_B20_final.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.4,0.4))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_B20_initial.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_B20_final.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0.5))










N = 100
B = 5
m = 95
mutation_pointsInf = np.linspace(0,m,int(m/5)+1)

## GENERATES THE ANALYTICAL PREDICTIONS.....................................................................................................................................................................
allLdistsInf, allLFdistsInf, allMdistsInf, allMFdistsInf = computeDists_infAlleles(m, B, numWs)
fitness_effects_valuesInf, DFEInf = computeDFE( allLdistsInf[-1], allMdistsInf[-1], wVector,lowest_fitness_contribution,highest_fitness_contribution)

mean_L_analyticalInf = np.zeros(len(mutation_pointsInf))
mean_M_analyticalInf = np.zeros(len(mutation_pointsInf))
mean_LF_analyticalInf = np.zeros(len(mutation_pointsInf))
mean_MF_analyticalInf = np.zeros(len(mutation_pointsInf))
for i in range(len(mutation_pointsInf)):
    mean_L_analyticalInf[i] = sp.quad(expectation,0.5,1.5,args=(allLdistsInf[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_M_analyticalInf[i] = sp.quad(expectation,0.5,1.5,args=(allMdistsInf[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_LF_analyticalInf[i] = sp.quad(expectation,0.5,1.5,args=(allLFdistsInf[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_MF_analyticalInf[i] = sp.quad(expectation,0.5,1.5,args=(allMFdistsInf[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    
analytical_fixation_timeInf = np.zeros(m+1)
analytical_mean_fitnessInf = np.zeros(m+1)
for i in range(m+1):
    analytical_fixation_timeInf[i] = 1/computeFixProb(allLdistsInf[i],allMdistsInf[i],computeFitness(allLdistsInf[i],lowest_fitness_contribution,highest_fitness_contribution),lowest_fitness_contribution,highest_fitness_contribution)
    analytical_mean_fitnessInf[i] = sp.quad(expectation,0.5,1.5,args=(allLdistsInf[int(i)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    





## GENERATES PLOTS WITH ANALYTICAL LINES AND SIMULATED HISTOGRAMS...........................................................................................................................................
fig, axs = plt.subplots(2,2,sharex=True,dpi=150, layout='constrained')
axs[1,1].plot([],[],color=(0.7,0.7,0.7))
axs[1,1].plot([],[],color=(0.3,0.3,0.3))
axs[0,0].set_ylabel("Probability density, $L_i$")
axs[0,1].set_ylabel("Probability density, $M_i$")
axs[1,0].set_xlabel('Fitness contribution, $w$')
axs[1,0].set_ylabel("Probability density, $L_{i|F}$")
axs[1,1].set_xlabel("Fitness contribution, $w'$ ")
axs[1,1].set_ylabel("Probability density, $M_{i|F}$")
axs[1,1].legend(('i = 0','i = 95'))

axs[0,0].plot(wVector,allLdistsInf[0],color=(0,0,1))
axs[0,0].plot(wVector,allLdistsInf[-1],color=(0,0,0.5))
axs[0,1].plot(wVector,allMdistsInf[0],color=(1,0,0))
axs[0,1].plot(wVector,allMdistsInf[-1],color=(0.5,0,0))
axs[1,0].plot(wVector,allLFdistsInf[0],color=(0,1,1))
axs[1,0].plot(wVector,allLFdistsInf[-1],color=(0,0.4,0.4))
axs[1,1].plot(wVector,allMFdistsInf[0],color=(1,0,1))
axs[1,1].plot(wVector,allMFdistsInf[-1],color=(0.5,0,0.5))

del allLFdistsInf
del allMFdistsInf

dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_BInf_initial.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_BInf_final.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,0.5))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_BInf_initial.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_BInf_final.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_BInf_initial.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,1,1))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_BInf_final.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.4,0.4))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_BInf_initial.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_BInf_final.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0.5))





















N = 100
B = 4
m = 60
mutation_points4 = np.linspace(0,m,int(m/5)+1)

## GENERATES THE ANALYTICAL PREDICTIONS.....................................................................................................................................................................
allLdists4, allLFdists4, allMdists4, allMFdists4 = computeDists(m, B, numWs)
fitness_effects_values4, DFE4 = computeDFE( allLdists4[-1], allMdists4[-1], wVector,lowest_fitness_contribution,highest_fitness_contribution)

mean_L_analytical4 = np.zeros(len(mutation_points4))
mean_M_analytical4 = np.zeros(len(mutation_points4))
mean_LF_analytical4 = np.zeros(len(mutation_points4))
mean_MF_analytical4 = np.zeros(len(mutation_points4))
for i in range(len(mutation_points4)):
    mean_L_analytical4[i] = sp.quad(expectation,0.5,1.5,args=(allLdists4[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_M_analytical4[i] = sp.quad(expectation,0.5,1.5,args=(allMdists4[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_LF_analytical4[i] = sp.quad(expectation,0.5,1.5,args=(allLFdists4[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_MF_analytical4[i] = sp.quad(expectation,0.5,1.5,args=(allMFdists4[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    
analytical_fixation_time4 = np.zeros(m+1)
analytical_mean_fitness4 = np.zeros(m+1)
for i in range(m+1):
    analytical_fixation_time4[i] = 1/computeFixProb(allLdists4[i],allMdists4[i],computeFitness(allLdists4[i],lowest_fitness_contribution,highest_fitness_contribution),lowest_fitness_contribution,highest_fitness_contribution)
    analytical_mean_fitness4[i] = sp.quad(expectation,0.5,1.5,args=(allLdists4[int(i)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    





## GENERATES PLOTS WITH ANALYTICAL LINES AND SIMULATED HISTOGRAMS...........................................................................................................................................
fig, axs = plt.subplots(2,2,sharex=True,dpi=150, layout='constrained')
axs[1,1].plot([],[],color=(0.7,0.7,0.7))
axs[1,1].plot([],[],color=(0.3,0.3,0.3))
axs[0,0].set_ylabel("Probability density, $L_i$")
axs[0,1].set_ylabel("Probability density, $M_i$")
axs[1,0].set_xlabel('Fitness contribution, $w$')
axs[1,0].set_ylabel("Probability density, $L_{i|F}$")
axs[1,1].set_xlabel("Fitness contribution, $w'$ ")
axs[1,1].set_ylabel("Probability density, $M_{i|F}$")
axs[1,1].legend(('i = 0','i = 60'))

axs[0,0].plot(wVector,allLdists4[0],color=(0,0,1))
axs[0,0].plot(wVector,allLdists4[-1],color=(0,0,0.5))
axs[0,1].plot(wVector,allMdists4[0],color=(1,0,0))
axs[0,1].plot(wVector,allMdists4[-1],color=(0.5,0,0))
axs[1,0].plot(wVector,allLFdists4[0],color=(0,1,1))
axs[1,0].plot(wVector,allLFdists4[-1],color=(0,0.4,0.4))
axs[1,1].plot(wVector,allMFdists4[0],color=(1,0,1))
axs[1,1].plot(wVector,allMFdists4[-1],color=(0.5,0,0.5))

del allLFdists4
del allMFdists4

dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B4_initial.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B4_final.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,0.5))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B4_initial.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B4_final.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_B4_initial.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,1,1))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_B4_final.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.4,0.4))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_B4_initial.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_B4_final.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0.5))











N = 100
B = 2
m = 35
mutation_points2 = np.linspace(0,m,int(m/5)+1)

## GENERATES THE ANALYTICAL PREDICTIONS.....................................................................................................................................................................
allLdists2, allLFdists2, allMdists2, allMFdists2 = computeDists(m, B, numWs)
fitness_effects_values2, DFE2 = computeDFE( allLdists2[-1], allMdists2[-1], wVector,lowest_fitness_contribution,highest_fitness_contribution)

mean_L_analytical2 = np.zeros(len(mutation_points2))
mean_M_analytical2 = np.zeros(len(mutation_points2))
mean_LF_analytical2 = np.zeros(len(mutation_points2))
mean_MF_analytical2 = np.zeros(len(mutation_points2))
for i in range(len(mutation_points2)):
    mean_L_analytical2[i] = sp.quad(expectation,0.5,1.5,args=(allLdists2[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_M_analytical2[i] = sp.quad(expectation,0.5,1.5,args=(allMdists2[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_LF_analytical2[i] = sp.quad(expectation,0.5,1.5,args=(allLFdists2[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    mean_MF_analytical2[i] = sp.quad(expectation,0.5,1.5,args=(allMFdists2[int(i*5)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    
analytical_fixation_time2 = np.zeros(m+1)
analytical_mean_fitness2 = np.zeros(m+1)
for i in range(m+1):
    analytical_fixation_time2[i] = 1/computeFixProb(allLdists2[i],allMdists2[i],computeFitness(allLdists2[i],lowest_fitness_contribution,highest_fitness_contribution),lowest_fitness_contribution,highest_fitness_contribution)
    analytical_mean_fitness2[i] = sp.quad(expectation,0.5,1.5,args=(allLdists2[int(i)],wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    





## GENERATES PLOTS WITH ANALYTICAL LINES AND SIMULATED HISTOGRAMS...........................................................................................................................................
fig, axs = plt.subplots(2,2,sharex=True,dpi=150, layout='constrained')
axs[1,1].plot([],[],color=(0.7,0.7,0.7))
axs[1,1].plot([],[],color=(0.3,0.3,0.3))
axs[0,0].set_ylabel("Probability density, $L_i$")
axs[0,1].set_ylabel("Probability density, $M_i$")
axs[1,0].set_xlabel('Fitness contribution, $w$')
axs[1,0].set_ylabel("Probability density, $L_{i|F}$")
axs[1,1].set_xlabel("Fitness contribution, $w'$ ")
axs[1,1].set_ylabel("Probability density, $M_{i|F}$")
axs[1,1].legend(('i = 0','i = 35'))

axs[0,0].plot(wVector,allLdists2[0],color=(0,0,1))
axs[0,0].plot(wVector,allLdists2[-1],color=(0,0,0.5))
axs[0,1].plot(wVector,allMdists2[0],color=(1,0,0))
axs[0,1].plot(wVector,allMdists2[-1],color=(0.5,0,0))
axs[1,0].plot(wVector,allLFdists2[0],color=(0,1,1))
axs[1,0].plot(wVector,allLFdists2[-1],color=(0,0.4,0.4))
axs[1,1].plot(wVector,allMFdists2[0],color=(1,0,1))
axs[1,1].plot(wVector,allMFdists2[-1],color=(0.5,0,0.5))

del allLFdists2
del allMFdists2

dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B2_initial.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B2_final.csv')
axs[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,0.5))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B2_initial.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B2_final.csv')
axs[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_B2_initial.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,1,1))
dataToPlot = np.loadtxt('data_multiplicative\Lfixdistribution_B2_final.csv')
axs[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.4,0.4))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_B2_initial.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mfixdistribution_B2_final.csv')
axs[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0.5,0,0.5))







fig2, axs2 = plt.subplots(2,4,dpi=200, layout='constrained',figsize=(12,5))
axs2[0,3].plot([],[],color=(0,0.5,0))
axs2[0,3].plot([],[],color=(1,0,0))
axs2[0,3].plot([],[],color=(0,0,1))
axs2[0,3].legend(["DFE","$L_i$","$M_i$"])
axs2[0,0].plot(fitness_effects_values2,DFE2,color=(0,0.5,0))
dataToPlot = np.loadtxt('data_multiplicative\DFE_B2.csv')
axs2[0,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.5,0))
axs2[0,1].plot(fitness_effects_values4,DFE4,color=(0,0.5,0))
dataToPlot = np.loadtxt('data_multiplicative\DFE_B4.csv')
axs2[0,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.5,0))
axs2[0,2].plot(fitness_effects_values20,DFE20,color=(0,0.5,0))
dataToPlot = np.loadtxt('data_multiplicative\DFE_B20.csv')
axs2[0,2].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.5,0))
axs2[0,3].plot(fitness_effects_valuesInf,DFEInf,color=(0,0.5,0))
dataToPlot = np.loadtxt('data_multiplicative\DFE_BInf.csv')
axs2[0,3].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0.5,0))

axs2[0,1].set_xlabel("Fitness effect, $s$")
axs2[0,1].xaxis.set_label_coords(1, -0.13)
axs2[0,0].set_ylabel("Probability density")


axs2[1,1].set_xlabel("Fitness contribution")
axs2[1,0].set_ylabel("Probability density")
axs2[1,1].xaxis.set_label_coords(1, -0.12)
axs2[0,0].set_title("$B=2$")
axs2[0,1].set_title("$B=4$")
axs2[0,2].set_title("$B=20$")
axs2[0,3].set_title("Infinite alleles")

#######
axs2[1,0].plot(wVector,allLdists2[-1],color=(0,0,1))
axs2[1,0].plot(wVector,allMdists2[-1],color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B2_final.csv')
axs2[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B2_final.csv')
axs2[1,0].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))
axs2[1,1].plot(wVector,allLdists4[-1],color=(0,0,1))
axs2[1,1].plot(wVector,allMdists4[-1],color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B4_final.csv')
axs2[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B4_final.csv')
axs2[1,1].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))
axs2[1,2].plot(wVector,allLdists20[-1],color=(0,0,1))
axs2[1,2].plot(wVector,allMdists20[-1],color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_B20_final.csv')
axs2[1,2].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_B20_final.csv')
axs2[1,2].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))
axs2[1,3].plot(wVector,allLdistsInf[-1],color=(0,0,1))
axs2[1,3].plot(wVector,allMdistsInf[-1],color=(1,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Ldistribution_BInf_final.csv')
axs2[1,3].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(0,0,1))
dataToPlot = np.loadtxt('data_multiplicative\Mdistribution_BInf_final.csv')
axs2[1,3].hist(dataToPlot,density=True,bins=100,alpha=0.5,color=(1,0,0))







fig, axs = plt.subplots(2,2,dpi=150, layout='constrained')
axs[0,0].plot([],[],'x',color="k")
axs[0,0].plot([],[],'+',color="k")
m = 35
mutation_points2 = np.linspace(0,m,int(m/5)+1)
dataToPlot = np.loadtxt('data_multiplicative\means_DFC_B2.csv')
axs[0,0].plot(mutation_points2,dataToPlot[0],'+',color=(0,0,1))
axs[0,0].plot(mutation_points2,dataToPlot[1],'+',color=(1,0,0))
axs[0,0].plot(mutation_points2,dataToPlot[2],'+',color=(0,1,1))
axs[0,0].plot(mutation_points2,dataToPlot[3],'+',color=(1,0,1))
axs[0,0].plot(mutation_points2,mean_L_analytical2,'x',color=(0,0,1))
axs[0,0].plot(mutation_points2,mean_M_analytical2,'x',color=(1,0,0))
axs[0,0].plot(mutation_points2,mean_LF_analytical2,'x',color=(0,1,1))
axs[0,0].plot(mutation_points2,mean_MF_analytical2,'x',color=(1,0,1))
axs[0,0].set_ylabel("Mean fitness contribution")


axs[0,1].plot([],[],'x',color="k")
axs[0,1].plot([],[],'+',color="k")
m = 60
mutation_points4 = np.linspace(0,m,int(m/5)+1)
dataToPlot = np.loadtxt('data_multiplicative\means_DFC_B4.csv')
axs[0,1].plot(mutation_points4,mean_L_analytical4,'x',color=(0,0,1))
axs[0,1].plot(mutation_points4,mean_M_analytical4,'x',color=(1,0,0))
axs[0,1].plot(mutation_points4,mean_LF_analytical4,'x',color=(0,1,1))
axs[0,1].plot(mutation_points4,mean_MF_analytical4,'x',color=(1,0,1))
axs[0,1].plot(mutation_points4,dataToPlot[0],'+',color=(0,0,1))
axs[0,1].plot(mutation_points4,dataToPlot[1],'+',color=(1,0,0))
axs[0,1].plot(mutation_points4,dataToPlot[2],'+',color=(0,1,1))
axs[0,1].plot(mutation_points4,dataToPlot[3],'+',color=(1,0,1))


axs[1,1].plot([],[],'x',color="k")
axs[1,1].plot([],[],'+',color="k")
m = 95
mutation_pointsInf = np.linspace(0,m,int(m/5)+1)
dataToPlot = np.loadtxt('data_multiplicative\means_DFC_BInf.csv')
axs[1,1].plot(mutation_pointsInf,mean_L_analyticalInf,'x',color=(0,0,1))
axs[1,1].plot(mutation_pointsInf,mean_M_analyticalInf,'x',color=(1,0,0))
axs[1,1].plot(mutation_pointsInf,mean_LF_analyticalInf,'x',color=(0,1,1))
axs[1,1].plot(mutation_pointsInf,mean_MF_analyticalInf,'x',color=(1,0,1))
axs[1,1].plot(mutation_pointsInf,dataToPlot[0],'+',color=(0,0,1))
axs[1,1].plot(mutation_pointsInf,dataToPlot[1],'+',color=(1,0,0))
axs[1,1].plot(mutation_pointsInf,dataToPlot[2],'+',color=(0,1,1))
axs[1,1].plot(mutation_pointsInf,dataToPlot[3],'+',color=(1,0,1))
axs[1,1].set_xlabel("Number of adaptive steps, $i$")
axs[1,1].legend(['Analytical','Simulations'], loc='lower right',fontsize=9)


axs[1,0].plot([],[],'x',color="k")
axs[1,0].plot([],[],'+',color="k")
m = 85
mutation_pointsInf = np.linspace(0,m,int(m/5)+1)
dataToPlot = np.loadtxt('data_multiplicative\means_DFC_B20.csv')
axs[1,0].plot(mutation_points20,mean_L_analytical20,'x',color=(0,0,1))
axs[1,0].plot(mutation_points20,mean_M_analytical20,'x',color=(1,0,0))
axs[1,0].plot(mutation_points20,mean_LF_analytical20,'x',color=(0,1,1))
axs[1,0].plot(mutation_points20,mean_MF_analytical20,'x',color=(1,0,1))
axs[1,0].plot(mutation_points20,dataToPlot[0],'+',color=(0,0,1))
axs[1,0].plot(mutation_points20,dataToPlot[1],'+',color=(1,0,0))
axs[1,0].plot(mutation_points20,dataToPlot[2],'+',color=(0,1,1))
axs[1,0].plot(mutation_points20,dataToPlot[3],'+',color=(1,0,1))
axs[1,0].set_xlabel("Number of adaptive steps, $i$")
axs[1,0].set_ylabel("Mean fitness contribution")











plt.figure(dpi=150)
plt.plot([],[],'.',color=(0,0,0))
plt.plot([],[],'.',color=(0.4,0.4,0))
plt.plot([],[],'.',color=(0.6,0.6,0))
plt.plot([],[],'.',color=(0.8,0.8,0))
plt.plot([],[],'x',color="k")
plt.plot([],[],'+',color="k")
dataToPlot = np.loadtxt('data_multiplicative\Fitness_over_time_BInf.csv')
plt.plot(dataToPlot[0],dataToPlot[1],"+",color=(0,0,0))
plt.plot(analytical_fixation_timeInf,analytical_mean_fitnessInf,"x",color=(0,0,0))
dataToPlot = np.loadtxt('data_multiplicative\Fitness_over_time_B20.csv')
plt.plot(dataToPlot[0],dataToPlot[1],"+",color=(0.4,0.4,0))
plt.plot(analytical_fixation_time20,analytical_mean_fitness20,"x",color=  (0.4,0.4,0))
dataToPlot = np.loadtxt('data_multiplicative\Fitness_over_time_B4.csv')
plt.plot(dataToPlot[0],dataToPlot[1],"+",color=(0.6,0.6,0))
plt.plot(analytical_fixation_time4,analytical_mean_fitness4,"x",color=(0.6,0.6,0))
dataToPlot = np.loadtxt('data_multiplicative\Fitness_over_time_B2.csv')
plt.plot(dataToPlot[0],dataToPlot[1],"+",color=(0.8,0.8,0))
plt.plot(analytical_fixation_time2,analytical_mean_fitness2,"x",color=(0.8,0.8,0))

plt.xlabel("Number of spontaneous mutations before fixation")
plt.ylabel("Mean fitness")
plt.legend(['Infinite alleles','$B=20$','$B=4$','$B=2$','Analytical','Simulations'])
plt.xscale('log')
plt.grid(True)
plt.xticks([3*10**2,10**3,3*10**3],labels=(str(3*10**2),str(10**3),str(3*10**3)))




overall_time_end = time.time()
print("Total time =",(overall_time_end-overall_time_start)/60," minutes")
