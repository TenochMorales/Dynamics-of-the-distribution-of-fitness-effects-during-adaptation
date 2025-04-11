import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
from numba.core.decorators import njit
import time
import numpy.random as rnd














#############
#############
############# Simulations for additive landscape
#############
#############




















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
            loc = rnd.randint(0,len(initialGenome))
            DFE[(B-1)*i+j] = (fitnessContributions[loc][np.mod(initialGenome[loc] + j+1,B)] - fitnessContributions[i][initialGenome[i]])/(len(initialGenome)*compute_fitness(initialGenome,fitnessContributions))
#            DFE[(B-1)*i+j] = (fitnessContributions[i][np.mod(initialGenome[i] + j+1,B)] - fitnessContributions[i][initialGenome[i]])/(len(initialGenome)*compute_fitness(initialGenome,fitnessContributions))
    return DFE



# .......... SELECTION AND OUTCOME OF MUTATION .......... ##
@njit
def fixation_probability_function(s,pop_size):
    return (1-np.exp(-2*s))/(1-np.exp(-2*pop_size*s))



@njit
def mutationTryAndOutcome(genome,fitnessContributions,B,pop_size):

    mutatedGenome, mutatedLoci = selectMutation(genome,B)

    s = -1 + compute_fitness(mutatedGenome,fitnessContributions)/compute_fitness(genome,fitnessContributions)
    success = False
    #if s > 0:
    if fixation_probability_function(s,pop_size) > rnd.rand():
        success = True
    return success, mutatedGenome





## .......... ADAPTIVE WALK .......... ##
@njit
def compute_addaptive_step(genome, fitnessContributions,B,pop_size):
    
    mutations = 0 
    tries = 0
        
    while mutations == 0:
        success, mutatedGenome = mutationTryAndOutcome(genome, fitnessContributions,B,pop_size)
        if success :  ## if a mutation happened
            genome = mutatedGenome.copy()
            mutations += 1
            #print("tries not surpassed")
        
        tries += 1
        if tries > 10**6:
            print('Maximum tries surpassed')
            genome = mutatedGenome.copy()
            mutations += 1
    
    return genome, tries




@njit
def samplingInWalks(N,B,numberWalks,numberMutations,pop_size):
    
    walk = 0
    samples_loci_distribution = np.zeros( (numberMutations+1,numberWalks*N) )
    samples_mutations_distribution = np.zeros( (numberMutations+1,numberWalks*N*(B-1)) )
    samples_fixed_loci_distribution = np.zeros( (numberMutations+1,numberWalks) )
    samples_fixed_mutations_distribution = np.zeros( (numberMutations+1,numberWalks) )
    mutations_until_fixation = np.zeros( (numberMutations+1,numberWalks) )
    samples_fixed_fitness_effects = np.zeros( (numberMutations+1,numberWalks) )
    
    samples_fitness_effects_distribution = np.zeros((numberMutations+1,numberWalks*N*(B-1)))

    while walk < numberWalks:
        
        
        
        if (walk*10/numberWalks) - int(walk*10/numberWalks) == 0:
            print("Walk: ",walk)
            
            
        initialGenome = np.zeros(N,dtype=np.int32)
        fitnessContributions = set_fitness_contributions(N,0,B)
        finalGenome = initialGenome.copy()
        for mutation in range(numberMutations + 1):
            
            beforeMutGenome = finalGenome.copy()
            finalGenome, tries = compute_addaptive_step(beforeMutGenome, fitnessContributions,B,pop_size)
            mutations_until_fixation[mutation][walk] = tries 
            L, M = computeDistributions(beforeMutGenome, fitnessContributions,B)
            for i in range(N):
                samples_loci_distribution[mutation][walk*N + i] = L[i]
                for a in range(B-1):
                    samples_mutations_distribution[mutation][(walk*N*(B-1) + i*(B-1) + a)] = M[i*(B-1) + a]
                    
                if beforeMutGenome[i] != finalGenome[i]:
                    samples_fixed_loci_distribution[mutation][walk] = fitnessContributions[i][beforeMutGenome[i]]
                    samples_fixed_mutations_distribution[mutation][walk] = fitnessContributions[i][finalGenome[i]]
                    samples_fixed_fitness_effects[mutation][walk] = (fitnessContributions[i][finalGenome[i]]-fitnessContributions[i][beforeMutGenome[i]])/(len(initialGenome)*compute_fitness(finalGenome, fitnessContributions))
                    
            DFE = compute_fitness_effects_distribution(beforeMutGenome,fitnessContributions,B)

            for i in range(len(DFE)):
                samples_fitness_effects_distribution[mutation][(walk*N*(B-1) + i)] = DFE[i]

        walk += 1

    return samples_loci_distribution, samples_mutations_distribution, samples_fixed_loci_distribution, samples_fixed_mutations_distribution, samples_fitness_effects_distribution, mutations_until_fixation, samples_fixed_fitness_effects










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
    
    pop_size = len(genome)*10
    mutatedGenome, mutatedLoci = selectMutation(genome,B)
    fitnessContributions[mutatedLoci][mutatedGenome[mutatedLoci]] = rnd.rand()
    s = -1 + compute_fitness(mutatedGenome,fitnessContributions)/compute_fitness(genome,fitnessContributions)
    success = False
    if fixation_probability_function(s,pop_size) > rnd.rand():
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
        if tries > 10**6:
            print('Maximum tries surpassed')
            mutations += 1
    
    return genome, tries, newFitnessContributions




@njit
def samplingInWalksInfAlleles(N,numberWalks,numberMutations,pop_size):
    B=10
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














totalNumberWalks = 10**3
numberWalks = 10**2
N = 100
B = 4
m = 200
pop_size = 10*N
timeStart = time.time()
file_for_saving = 'data_additive_SI'

mutation_points20 = np.linspace(0,m,int(m/5)+1)
mean_L_simulations20 = np.zeros(len(mutation_points20))
mean_M_simulations20 = np.zeros(len(mutation_points20))
mean_LF_simulations20 = np.zeros(len(mutation_points20))
mean_MF_simulations20  = np.zeros(len(mutation_points20))
simulation_fixation_time20 = np.zeros(m+1)
simulation_mean_fitness20 = np.zeros(m+1)

## GENERATES THE SIMULATION FILES TO BE READ LATER..........................................................................................................................................................
for run in range(int(totalNumberWalks/numberWalks)):
    print(run)
    L20, M20, Lfixed20, Mfixed20, DFEsamples20, fixationTime20, DFEfixed20 = samplingInWalks(N,B,numberWalks,m,pop_size)
    #L20, M20, Lfixed20, Mfixed20, DFEsamples20, fixationTime20 = samplingInWalksInfAlleles(N,numberWalks,m,pop_size)
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

    number_boxplots = 11
    mut_boxplot = np.arange(number_boxplots)
    mut_boxplot = mut_boxplot*int(m/(number_boxplots-1))
    data_L_boxplots = np.zeros((len(L20[0]),number_boxplots))
    data_M_boxplots = np.zeros((len(M20[0]),number_boxplots))
    data_LF_boxplots = np.zeros((len(Lfixed20[0]),number_boxplots))
    data_MF_boxplots = np.zeros((len(Mfixed20[0]),number_boxplots))
    data_DFE_boxplots = np.zeros((len(DFEsamples20[0]),number_boxplots))
    data_DFE_fixed_boxplots = np.zeros((len(DFEfixed20[0]),number_boxplots))


    for i in range(number_boxplots):
        data_L_boxplots[:,i] = L20[mut_boxplot[i]]
        data_M_boxplots[:,i] = M20[mut_boxplot[i]]
        data_LF_boxplots[:,i] = Lfixed20[mut_boxplot[i]]
        data_MF_boxplots[:,i] = Mfixed20[mut_boxplot[i]]
        data_DFE_boxplots[:,i] = DFEsamples20[mut_boxplot[i]]
        data_DFE_fixed_boxplots[:,i] = DFEfixed20[mut_boxplot[i]]


    with open(file_for_saving+"\Ldistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv",'a') as f:
        np.savetxt(f,data_L_boxplots)    
    with open(file_for_saving+"\Mdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv",'a') as f:
        np.savetxt(f,data_M_boxplots)  
    with open(file_for_saving+"\LFdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv",'a') as f:
        np.savetxt(f,data_LF_boxplots)  
    with open(file_for_saving+"\MFdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv",'a') as f:
        np.savetxt(f,data_MF_boxplots)  
    with open(file_for_saving+"\DFEdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv",'a') as f:
        np.savetxt(f,data_DFE_boxplots)
        
  
        
    for i in range(len(mutation_points20)):
        mean_L_simulations20[i] += np.mean(L20[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_M_simulations20[i] += np.mean(M20[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_LF_simulations20[i] += np.mean(Lfixed20[int(i*5)])/(totalNumberWalks/numberWalks)
        mean_MF_simulations20[i] += np.mean(Mfixed20[int(i*5)])/(totalNumberWalks/numberWalks)

    for i in range(m+1):
        simulation_fixation_time20[i] += np.mean(fixationTime20[i])/(totalNumberWalks/numberWalks)
        simulation_mean_fitness20[i] += np.mean(L20[i])/(totalNumberWalks/numberWalks)
    



boxprops_L = dict(linestyle='-', linewidth=1, color=[0,0,0.8])
meanprops_L =dict(linestyle='-', linewidth=1, color=[0,0,0.8])
whiskerprops_L =dict(linestyle='-', linewidth=1, color=[0,0,0.8])
boxprops_M = dict(linestyle='-', linewidth=1, color=[0.8,0,0])
meanprops_M =dict(linestyle='-', linewidth=1, color=[0.8,0,0])
whiskerprops_M =dict(linestyle='-', linewidth=1, color=[0.8,0,0])
boxprops_LF = dict(linestyle='-', linewidth=1, color=[0,0.8,0.8])
meanprops_LF =dict(linestyle='-', linewidth=1, color=[0,0.8,0.8])
whiskerprops_LF =dict(linestyle='-', linewidth=1, color=[0,0.8,0.8])
boxprops_MF = dict(linestyle='-', linewidth=1, color=[0.8,0,0.8])
meanprops_MF =dict(linestyle='-', linewidth=1, color=[0.8,0,0.8])
whiskerprops_MF =dict(linestyle='-', linewidth=1, color=[0.8,0,0.8])
plt.figure(dpi=150)
plt.boxplot(data_L_boxplots,positions=mut_boxplot,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_L,meanprops=meanprops_L,whiskerprops=whiskerprops_L)
plt.boxplot(data_M_boxplots,positions=mut_boxplot+m/200,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_M,meanprops=meanprops_M,whiskerprops=whiskerprops_M)
plt.xticks(mut_boxplot)
plt.figure(dpi=150)
plt.boxplot(data_LF_boxplots,positions=mut_boxplot,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_LF,meanprops=meanprops_LF,whiskerprops=whiskerprops_LF)
plt.boxplot(data_MF_boxplots,positions=mut_boxplot+m/200,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_MF,meanprops=meanprops_MF,whiskerprops=whiskerprops_MF)
plt.xticks(mut_boxplot)

plt.figure(dpi=150)
plt.boxplot(data_DFE_fixed_boxplots,positions=mut_boxplot,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_LF,meanprops=meanprops_LF,whiskerprops=whiskerprops_LF)
plt.grid(True)
plt.figure(dpi=150)
plt.hist(data_DFE_fixed_boxplots[:,-1],density=True,bins=50)