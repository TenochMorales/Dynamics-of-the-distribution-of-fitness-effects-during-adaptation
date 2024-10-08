import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
from numba.core.decorators import njit
import time


@njit
def computeFitnessFunc(w,Ldist,wVector):
    index = np.argmin(np.abs(wVector - w))
    return w*Ldist[index]
    
def computeFitness(Ldist,wVector):
    
    result = sp.quad(computeFitnessFunc, 0, 1, args=(Ldist,wVector),limit = 100000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    return result



# Gives integrand in equation 6 from overleaf (and so also equation 5)
@njit
def LFixFunc(wPrime, w, Ldist, Mdist,Fitness,wVector):
    index = np.argmin(np.abs(wVector - wPrime))
    fVal = 2 * ((wPrime - w) / (N * Fitness))
    return Mdist[index] * fVal



# Gives integrand in equation 7 from overleaf
@njit
def MFixFunc(w,wPrime,Ldist,Mdist,Fitness,wVector):
    index = np.argmin(np.abs(wVector - w))
    fVal = 2 * ((wPrime - w) / (N * Fitness))

    return Ldist[index] * fVal



def fixProbFunc(w, Ldist, Mdist,Fitness,wVector):
    index = np.argmin(np.abs(wVector-w))
    integrand = sp.quad(MFixFunc, 0, w, args=(w,Ldist,Mdist,Fitness,wVector),limit = 100000,epsabs=1.0e-04, epsrel=1.0e-04)[0] # w, 1
    return Mdist[index]*integrand



# Gives equation 5 in overleaf
def computeFixProb(Ldist, Mdist,Fitness,wVector):
    result = sp.quad(fixProbFunc,0, 1, args=(Ldist,Mdist,Fitness,wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0] # 0, 1
    return result



# Gives equation 6 in overleaf
def makeLFdist(Ldist, Mdist,FProb, Fitness,wVector):
    LFdist = np.zeros(len(Ldist))
    for i in range(len(LFdist)):
        LFdist[i] = (Ldist[i]/FProb)*sp.quad(LFixFunc, wVector[i], 1, args=(wVector[i], Ldist, Mdist, Fitness,wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0] # wVector[i], 1

    return LFdist


# Gives equation 7 in overleaf
def makeMFdist(Ldist, Mdist, FProb,Fitness,wVector):

    MFdist = np.zeros(len(Mdist))
    for i in range(len(Mdist)):
        MFdist[i] = (Mdist[i]/FProb)*sp.quad(MFixFunc, 0, wVector[i], args=(wVector[i], Ldist, Mdist,Fitness,wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]

    return MFdist


# Gives equation 9 in overleaf
@njit
def nextL(Ldist,LFdist,MFdist,wVector):
    nextLdist = np.zeros(len(Ldist))
    for i in range(len(Ldist)):
        nextLdist[i] = Ldist[i] + MFdist[i]/N - LFdist[i]/N

    return nextLdist


# Gives equation 10 in overleaf
@njit
def nextM(Mdist, LFdist, MFdist,B,numWs,wVector):
    nextMdist = np.zeros(len(Mdist))
    for i in range(len(Mdist)):
        nextMdist[i] = Mdist[i] + LFdist[i]/((B-1)*N) - MFdist[i]/((B-1)*N)

    return nextMdist



############################################################
def computeDists(m,B,numWs):

    allLdists = np.zeros((m + 1, numWs))
    allMdists = np.zeros((m + 1, numWs))
    allLFdists = np.zeros((m + 1, numWs))
    allMFdists = np.zeros((m + 1, numWs))
    
    initialLdist = np.zeros(numWs)
    initialMdist = np.zeros(numWs)

    ## DIFFERENT INITAL DISTS (comment out to just use uniform) ##
    i = 0
    wVector = np.linspace(0, 1, num = numWs)
    for w in wVector:
        initialLdist[i] = 1#2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        initialMdist[i] = 1#2-2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        i = i + 1
        
    Ldist = initialLdist
    Mdist = initialMdist


    for i in range(m + 1):

        Fitness = computeFitness(Ldist,wVector)
        FProb = computeFixProb(Ldist, Mdist,Fitness,wVector)
        print("Mutation =",i,"of",str(m))

        allLdists[i] = Ldist
        allMdists[i] = Mdist

        LFdist = makeLFdist(Ldist, Mdist, FProb, Fitness,wVector)
        MFdist = makeMFdist(Ldist, Mdist, FProb, Fitness,wVector)

        allLFdists[i] = LFdist
        allMFdists[i] = MFdist

        Ldist = nextL(Ldist, LFdist, MFdist,wVector)
        Mdist = nextM(Mdist, LFdist, MFdist,B,numWs,wVector)


    return allLdists, allLFdists, allMdists, allMFdists



def expectation(w,dist,wVector):
    index = np.argmin(np.abs(np.array(wVector) - w))
    pdf = dist[index]
    return w*pdf


def variance(w,dist,mean,wVector):
    
    index = np.argmin(np.abs(np.array(wVector) - w))
    pdf = dist[index]
    return (w-mean)**2*pdf


def fitness_over_time(Ldist_vector,Mdist_vector,wVector):
    time_vector = np.zeros(len(Ldist_vector))
    fitness_vector = np.zeros(len(Ldist_vector))
    fitness_vector[0] = computeFitness(Ldist_vector[0],wVector)
    for i in range(1,len(Ldist_vector)):
        fitness_vector[i] = computeFitness(Ldist_vector[i],wVector)
        time_vector[i] = time_vector[i-1] + 1/computeFixProb(Ldist_vector[i],Mdist_vector[i],fitness_vector[i],wVector)
    
    return np.array([time_vector,fitness_vector])


def check_normalizationFunc(w,distribution,wVector):
    
    index = np.argmin(np.abs(np.array(wVector) - w))
    pdf = distribution[index]
    return pdf

def check_normalization(distribution,wVector):
    
    result = sp.quad(check_normalizationFunc, 0, 1, args=(distribution,wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    return result





def integrand_dfe(w,s,Ldist,Mdist,wVector,W):
    
    Lindex = np.argmin(np.abs(np.array(wVector)-w))
    L = Ldist[Lindex]

    wPrime = N*s*W+w
    Mindex = np.argmin(np.abs(np.array(wVector)-wPrime))
    M = Mdist[Mindex]

    return N*L*M*W


def computeDFE(Ldist,Mdist,wVector):
    
    
    W = sp.quad(expectation,0,1,args=(Ldist,wVector),limit = 10000,epsabs=1.0e-04, epsrel=1.0e-04)[0]
    sDist = np.linspace(-1/(N*W),1/(N*W), num =len(wVector)) # sDist = np.linspace(-1/(N*W), 1/(N*W), num = 1000)
    DFE = np.zeros(len(sDist))
    DFE_higher_abs_error = 0
    i = 0
    for s in sDist:
        if s <=0:
            lowBound = -N*s*W
            val = sp.quad(integrand_dfe,lowBound,1,args = (s,Ldist,Mdist,wVector,W),limit = 10000,epsabs=1.0e-03, epsrel=1.0e-03, points=[0]) # lowBound, 1
        elif s > 0:
            uppBound = 1 - N*s*W
            val = sp.quad(integrand_dfe,0,uppBound,args = (s,Ldist,Mdist,wVector,W),limit = 10000,epsabs=1.0e-03, epsrel=1.0e-03,points=[0]) # 0, uppBound
    
        DFE[i]+=val[0]      
        if val[1] > DFE_higher_abs_error:
            DFE_higher_abs_error = val[1]
            #print("New higher abs error =",DFE_higher_abs_error)
        i+=1
        
    return sDist, DFE

















#############
#############
############# Simulations for additive landscape
#############
#############






















import numpy.random as rnd
from numba.core.decorators import njit







@njit
def set_fitness_contributions(numberLoci, initialDistribution,B):

    fitnessContributions = np.zeros((numberLoci,B))
    for i in range(numberLoci):
        for j in range(B):
            fitnessContributions[i][j] = rnd.rand()

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
    
    Fitness = 0
    for i in range(len(genome)):
        Fitness += fitnessContributions[i][genome[i]]

    return Fitness/len(genome)








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
            DFE[(B-1)*i+j] = (fitnessContributions[i][np.mod(initialGenome[i] + j+1,B)] - fitnessContributions[i][initialGenome[i]])/(len(initialGenome)*compute_fitness(initialGenome,fitnessContributions))

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
            genome = mutatedGenome.copy()
            mutations += 1
    
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
def nextM_infAlleles(Mdist, LFdist, MFdist,B,numWs,wVector):
    nextM_infAllelesdist = np.zeros(len(Mdist))
    for i in range(len(Mdist)):
        nextM_infAllelesdist[i] = Mdist[i] #+ LFdist[i]/((B-1)*N) - MFdist[i]/((B-1)*N)

    return nextM_infAllelesdist



############################################################
def computeDists_infAlleles(m,B,numWs):

    allLdists = np.zeros((m + 1, numWs))
    allMdists = np.zeros((m + 1, numWs))
    allLFdists = np.zeros((m + 1, numWs))
    allMFdists = np.zeros((m + 1, numWs))
    
    initialLdist = np.zeros(numWs)
    initialMdist = np.zeros(numWs)

    ## DIFFERENT INITAL DISTS (comment out to just use uniform) ##
    i = 0
    wVector = np.linspace(0, 1, num = numWs)
    for w in wVector:
        initialLdist[i] = 1#2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        initialMdist[i] = 1#2-2*w #np.exp(4)/(np.exp(4)-1)*np.exp(-w)
        i = i + 1
        
    Ldist = initialLdist
    Mdist = initialMdist


    for i in range(m + 1):

        Fitness = computeFitness(Ldist,wVector)
        FProb = computeFixProb(Ldist, Mdist,Fitness,wVector)
        print("Mutation =",i,"of",str(m))

        allLdists[i] = Ldist
        allMdists[i] = Mdist

        LFdist = makeLFdist(Ldist, Mdist, FProb, Fitness,wVector)
        MFdist = makeMFdist(Ldist, Mdist, FProb, Fitness,wVector)

        allLFdists[i] = LFdist
        allMFdists[i] = MFdist

        Ldist = nextL(Ldist, LFdist, MFdist,wVector)
        Mdist = nextM_infAlleles(Mdist, LFdist, MFdist,B,numWs,wVector)



    return allLdists, allLFdists, allMdists, allMFdists




# .......... SELECTION AND OUTCOME OF MUTATION .......... ##
@njit
def mutationTryAndOutcome_inf_alleles(genome,fitnessContributions,B):

    mutatedGenome, mutatedLoci = selectMutation(genome,B)
    fitnessContributions[mutatedLoci][mutatedGenome[mutatedLoci]] = rnd.rand()
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














totalNumberWalks = 10**5
numberWalks = 10**4


N = 100
B = 20
m = 85
timeStart = time.time()

mutation_points20 = np.linspace(0,m,int(m/5)+1)
mean_L_simulations20 = np.zeros(len(mutation_points20))
mean_M_simulations20 = np.zeros(len(mutation_points20))
mean_LF_simulations20 = np.zeros(len(mutation_points20))
mean_MF_simulations20  = np.zeros(len(mutation_points20))
simulation_fixation_time20 = np.zeros(m+1)
simulation_mean_fitness20 = np.zeros(m+1)

## GENERATES THE SIMULATION FILES TO BE READ LATER..........................................................................................................................................................
for run in range(int(totalNumberWalks/numberWalks)):
    L20, M20, Lfixed20, Mfixed20, DFEsamples20, fixationTime20 = samplingInWalks(N,B,numberWalks,m)
    initial_beneficial_fraction = 0
    
    for i in range(len(DFEsamples20[0])):
        if DFEsamples20[0][i] > 0:
            initial_beneficial_fraction += 1
    print('initial_beneficial_fraction =', initial_beneficial_fraction/len(DFEsamples20[0]))
    final_beneficial_fraction = 0
    for i in range(len(DFEsamples20[0])):
        if DFEsamples20[-1][i] > 0:
            final_beneficial_fraction += 1
    print('final_beneficial_fraction =', final_beneficial_fraction/len(DFEsamples20[0]))

    with open("data_additive\Ldistribution_B20_initial.csv",'a') as f:
        np.savetxt(f,L20[0])  
    with open("data_additive\Mdistribution_B20_initial.csv",'a') as f:
        np.savetxt(f,M20[0])  
    with open("data_additive\Lfixdistribution_B20_initial.csv",'a') as f:
        np.savetxt(f,Lfixed20[0])  
    with open("data_additive\Mfixdistribution_B20_initial.csv",'a') as f:
        np.savetxt(f,Mfixed20[0])  
    with open("data_additive\Ldistribution_B20_final.csv",'a') as f:
        np.savetxt(f,L20[-1])  
    with open("data_additive\Mdistribution_B20_final.csv",'a') as f:
        np.savetxt(f,M20[-1])  
    with open("data_additive\Lfixdistribution_B20_final.csv",'a') as f:
        np.savetxt(f,Lfixed20[-1])  
    with open("data_additive\Mfixdistribution_B20_final.csv",'a') as f:
        np.savetxt(f,Mfixed20[-1]) 
    with open("data_additive\DFE_B20.csv",'a') as f:
        np.savetxt(f,DFEsamples20[-1])     
        
    for i in range(len(mutation_points20)):
        mean_L_simulations20[i] += np.mean(L20[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_M_simulations20[i] += np.mean(M20[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_LF_simulations20[i] += np.mean(Lfixed20[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_MF_simulations20[i] += np.mean(Mfixed20[int(i*5)])/(totalNumberWalks/numberWalks)

    for i in range(m+1):
        simulation_fixation_time20[i] += np.mean(fixationTime20[i])/(totalNumberWalks/numberWalks)
        simulation_mean_fitness20[i] += np.mean(L20[i])/(totalNumberWalks/numberWalks)

    del L20
    del M20
    del Lfixed20
    del Mfixed20
    del DFEsamples20
with open("data_additive\means_DFC_B20.csv",'a') as f:
    
    dataToSave = np.zeros((4,len(mean_L_simulations20)))
    dataToSave[0] = mean_L_simulations20
    dataToSave[1] = mean_M_simulations20
    dataToSave[2] = mean_LF_simulations20
    dataToSave[3] = mean_MF_simulations20
    np.savetxt(f,dataToSave)
    
with open("data_additive\Fitness_over_time_B20.csv",'a') as f:
    dataToSave = np.zeros((2,len(simulation_fixation_time20)))
    dataToSave[0] = simulation_fixation_time20
    dataToSave[1] = simulation_mean_fitness20
    np.savetxt(f,dataToSave)



N = 100
B = 5
m = 95
timeStart = time.time()

mutation_pointsInf = np.linspace(0,m,int(m/5)+1)
mean_L_simulationsInf = np.zeros(len(mutation_pointsInf))
mean_M_simulationsInf = np.zeros(len(mutation_pointsInf))
mean_LF_simulationsInf = np.zeros(len(mutation_pointsInf))
mean_MF_simulationsInf  = np.zeros(len(mutation_pointsInf))
simulation_fixation_timeInf = np.zeros(m+1)
simulation_mean_fitnessInf = np.zeros(m+1)

## GENERATES THE SIMULATION FILES TO BE READ LATER..........................................................................................................................................................
for run in range(int(totalNumberWalks/numberWalks)):
    LInf, MInf, LfixedInf, MfixedInf, DFEsamplesInf, fixationTimeInf = samplingInWalksInfAlleles(N,B,numberWalks,m)
    initial_beneficial_fraction = 0
    
    for i in range(len(DFEsamplesInf[0])):
        if DFEsamplesInf[0][i] > 0:
            initial_beneficial_fraction += 1
    print('initial_beneficial_fraction =', initial_beneficial_fraction/len(DFEsamplesInf[0]))
    final_beneficial_fraction = 0
    for i in range(len(DFEsamplesInf[0])):
        if DFEsamplesInf[-1][i] > 0:
            final_beneficial_fraction += 1
    print('final_beneficial_fraction =', final_beneficial_fraction/len(DFEsamplesInf[0]))

    with open("data_additive\Ldistribution_BInf_initial.csv",'a') as f:
        np.savetxt(f,LInf[0])  
    with open("data_additive\Mdistribution_BInf_initial.csv",'a') as f:
        np.savetxt(f,MInf[0])  
    with open("data_additive\Lfixdistribution_BInf_initial.csv",'a') as f:
        np.savetxt(f,LfixedInf[0])  
    with open("data_additive\Mfixdistribution_BInf_initial.csv",'a') as f:
        np.savetxt(f,MfixedInf[0])  
    with open("data_additive\Ldistribution_BInf_final.csv",'a') as f:
        np.savetxt(f,LInf[-1])  
    with open("data_additive\Mdistribution_BInf_final.csv",'a') as f:
        np.savetxt(f,MInf[-1])  
    with open("data_additive\Lfixdistribution_BInf_final.csv",'a') as f:
        np.savetxt(f,LfixedInf[-1])  
    with open("data_additive\Mfixdistribution_BInf_final.csv",'a') as f:
        np.savetxt(f,MfixedInf[-1]) 
    with open("data_additive\DFE_BInf.csv",'a') as f:
        np.savetxt(f,DFEsamplesInf[-1])     
        
    for i in range(len(mutation_pointsInf)):
        mean_L_simulationsInf[i] += np.mean(LInf[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_M_simulationsInf[i] += np.mean(MInf[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_LF_simulationsInf[i] += np.mean(LfixedInf[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_MF_simulationsInf[i] += np.mean(MfixedInf[int(i*5)])/(totalNumberWalks/numberWalks)

    for i in range(m+1):
        simulation_fixation_timeInf[i] += np.mean(fixationTimeInf[i])/(totalNumberWalks/numberWalks)
        simulation_mean_fitnessInf[i] += np.mean(LInf[i])/(totalNumberWalks/numberWalks)            
    del LInf
    del MInf
    del LfixedInf
    del MfixedInf
    del DFEsamplesInf

with open("data_additive\means_DFC_BInf.csv",'a') as f:
   
   dataToSave = np.zeros((4,len(mean_L_simulationsInf)))
   dataToSave[0] = mean_L_simulationsInf
   dataToSave[1] = mean_M_simulationsInf
   dataToSave[2] = mean_LF_simulationsInf
   dataToSave[3] = mean_MF_simulationsInf
   np.savetxt(f,dataToSave)
   
with open("data_additive\Fitness_over_time_BInf.csv",'a') as f:
   dataToSave = np.zeros((2,len(simulation_fixation_timeInf)))
   dataToSave[0] = simulation_fixation_timeInf
   dataToSave[1] = simulation_mean_fitnessInf
   np.savetxt(f,dataToSave)







N = 100
B = 4
m = 60
timeStart = time.time()

mutation_points4 = np.linspace(0,m,int(m/5)+1)
mean_L_simulations4 = np.zeros(len(mutation_points4))
mean_M_simulations4 = np.zeros(len(mutation_points4))
mean_LF_simulations4 = np.zeros(len(mutation_points4))
mean_MF_simulations4  = np.zeros(len(mutation_points4))
simulation_fixation_time4 = np.zeros(m+1)
simulation_mean_fitness4 = np.zeros(m+1)

## GENERATES THE SIMULATION FILES TO BE READ LATER..........................................................................................................................................................
for run in range(int(totalNumberWalks/numberWalks)):
    L4, M4, Lfixed4, Mfixed4, DFEsamples4, fixationTime4 = samplingInWalks(N,B,numberWalks,m)
    initial_beneficial_fraction = 0
    
    for i in range(len(DFEsamples4[0])):
        if DFEsamples4[0][i] > 0:
            initial_beneficial_fraction += 1
    print('initial_beneficial_fraction =', initial_beneficial_fraction/len(DFEsamples4[0]))
    final_beneficial_fraction = 0
    for i in range(len(DFEsamples4[0])):
        if DFEsamples4[-1][i] > 0:
            final_beneficial_fraction += 1
    print('final_beneficial_fraction =', final_beneficial_fraction/len(DFEsamples4[0]))

    with open("data_additive\Ldistribution_B4_initial.csv",'a') as f:
        np.savetxt(f,L4[0])  
    with open("data_additive\Mdistribution_B4_initial.csv",'a') as f:
        np.savetxt(f,M4[0])  
    with open("data_additive\Lfixdistribution_B4_initial.csv",'a') as f:
        np.savetxt(f,Lfixed4[0])  
    with open("data_additive\Mfixdistribution_B4_initial.csv",'a') as f:
        np.savetxt(f,Mfixed4[0])  
    with open("data_additive\Ldistribution_B4_final.csv",'a') as f:
        np.savetxt(f,L4[-1])  
    with open("data_additive\Mdistribution_B4_final.csv",'a') as f:
        np.savetxt(f,M4[-1])  
    with open("data_additive\Lfixdistribution_B4_final.csv",'a') as f:
        np.savetxt(f,Lfixed4[-1])  
    with open("data_additive\Mfixdistribution_B4_final.csv",'a') as f:
        np.savetxt(f,Mfixed4[-1]) 
    with open("data_additive\DFE_B4.csv",'a') as f:
        np.savetxt(f,DFEsamples4[-1])     
        
    for i in range(len(mutation_points4)):
        mean_L_simulations4[i] += np.mean(L4[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_M_simulations4[i] += np.mean(M4[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_LF_simulations4[i] += np.mean(Lfixed4[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_MF_simulations4[i] += np.mean(Mfixed4[int(i*5)])/(totalNumberWalks/numberWalks)

    for i in range(m+1):
        simulation_fixation_time4[i] += np.mean(fixationTime4[i])/(totalNumberWalks/numberWalks)
        simulation_mean_fitness4[i] += np.mean(L4[i])/(totalNumberWalks/numberWalks)
    
        
    del L4
    del M4
    del Lfixed4
    del Mfixed4
    del DFEsamples4
    
with open("data_additive\means_DFC_B4.csv",'a') as f:
        
    dataToSave = np.zeros((4,len(mean_L_simulations4)))
    dataToSave[0] = mean_L_simulations4
    dataToSave[1] = mean_M_simulations4
    dataToSave[2] = mean_LF_simulations4
    dataToSave[3] = mean_MF_simulations4
    np.savetxt(f,dataToSave)
        
with open("data_additive\Fitness_over_time_B4.csv",'a') as f:
    dataToSave = np.zeros((2,len(simulation_fixation_time4)))
    dataToSave[0] = simulation_fixation_time4
    dataToSave[1] = simulation_mean_fitness4
    np.savetxt(f,dataToSave)




N = 100
B = 2
m = 35
timeStart = time.time()

mutation_points2 = np.linspace(0,m,int(m/5)+1)
mean_L_simulations2 = np.zeros(len(mutation_points2))
mean_M_simulations2 = np.zeros(len(mutation_points2))
mean_LF_simulations2 = np.zeros(len(mutation_points2))
mean_MF_simulations2  = np.zeros(len(mutation_points2))
simulation_fixation_time2 = np.zeros(m+1)
simulation_mean_fitness2 = np.zeros(m+1)

## GENERATES THE SIMULATION FILES TO BE READ LATER..........................................................................................................................................................
for run in range(int(totalNumberWalks/numberWalks)):
    L2, M2, Lfixed2, Mfixed2, DFEsamples2, fixationTime2 = samplingInWalks(N,B,numberWalks,m)
    initial_beneficial_fraction = 0
    
    for i in range(len(DFEsamples2[0])):
        if DFEsamples2[0][i] > 0:
            initial_beneficial_fraction += 1
    print('initial_beneficial_fraction =', initial_beneficial_fraction/len(DFEsamples2[0]))
    final_beneficial_fraction = 0
    for i in range(len(DFEsamples2[0])):
        if DFEsamples2[-1][i] > 0:
            final_beneficial_fraction += 1
    print('final_beneficial_fraction =', final_beneficial_fraction/len(DFEsamples2[0]))

    with open("data_additive\Ldistribution_B2_initial.csv",'a') as f:
        np.savetxt(f,L2[0])  
    with open("data_additive\Mdistribution_B2_initial.csv",'a') as f:
        np.savetxt(f,M2[0])  
    with open("data_additive\Lfixdistribution_B2_initial.csv",'a') as f:
        np.savetxt(f,Lfixed2[0])  
    with open("data_additive\Mfixdistribution_B2_initial.csv",'a') as f:
        np.savetxt(f,Mfixed2[0])  
    with open("data_additive\Ldistribution_B2_final.csv",'a') as f:
        np.savetxt(f,L2[-1])  
    with open("data_additive\Mdistribution_B2_final.csv",'a') as f:
        np.savetxt(f,M2[-1])  
    with open("data_additive\Lfixdistribution_B2_final.csv",'a') as f:
        np.savetxt(f,Lfixed2[-1])  
    with open("data_additive\Mfixdistribution_B2_final.csv",'a') as f:
        np.savetxt(f,Mfixed2[-1]) 
    with open("data_additive\DFE_B2.csv",'a') as f:
        np.savetxt(f,DFEsamples2[-1])     
        
    for i in range(len(mutation_points2)):
        mean_L_simulations2[i] += np.mean(L2[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_M_simulations2[i] += np.mean(M2[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_LF_simulations2[i] += np.mean(Lfixed2[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_MF_simulations2[i] += np.mean(Mfixed2[int(i*5)])/(totalNumberWalks/numberWalks)

    for i in range(m+1):
        simulation_fixation_time2[i] += np.mean(fixationTime2[i])/(totalNumberWalks/numberWalks)
        simulation_mean_fitness2[i] += np.mean(L2[i])/(totalNumberWalks/numberWalks)
    
        
    del L2
    del M2
    del Lfixed2
    del Mfixed2
    del DFEsamples2

with open("data_additive\means_DFC_B2.csv",'a') as f:
    dataToSave = np.zeros((4,len(mean_L_simulations2)))
    dataToSave[0] = mean_L_simulations2
    dataToSave[1] = mean_M_simulations2
    dataToSave[2] = mean_LF_simulations2
    dataToSave[3] = mean_MF_simulations2
    np.savetxt(f,dataToSave)
        
with open("data_additive\Fitness_over_time_B2.csv",'a') as f:
    dataToSave = np.zeros((2,len(simulation_fixation_time2)))
    dataToSave[0] = simulation_fixation_time2
    dataToSave[1] = simulation_mean_fitness2
    np.savetxt(f,dataToSave)







