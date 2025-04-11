import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
from numba.core.decorators import njit
import time
import numpy.random as rnd
import scipy.interpolate as si
import matplotlib as mpl




























































# ANALYTICAL
@njit
def chebishev_interpolation(x,x_points,y_points):
    '''
    Generates the interpolation of the function f(x)
    at the desired x-value for the list of
    ordered points (x_points,y_points).
    Returns the value f(x) for some x approximated by a 
    polynomial of degree len(x_points)-1
    '''
    n = len(x_points)
    coef = np.zeros((n,n)) ## Newton's coefficient matrix
    for i in range(n):
        coef[i,0] = y_points[i]
    for j in range(1,n):
        for i in range(n-j):
            coef[i,j] = (coef[i+1,j-1]-coef[i,j-1])/(x_points[i+j]-x_points[i])
    poly = coef[0,-1]
    for k in range(1,n):
        poly = coef[0,n-1-k] + (x - x_points[n-1-k])*poly
    return poly
@njit
def fixation_prob_function(w,x,N,Fitness,pop_size):
    s = (x-w)/(N*Fitness)
    if abs(s) < 10**(-8):
        result = 1/(pop_size)
    else:
        result = (1-np.exp(-2*s))/(1-np.exp(-2*pop_size*s))
    return result
@njit
def fixed_L_integrand(x, w, M_dist, x_vector,N,Fitness,pop_size, interpolation_function):
    fix_prob =  fixation_prob_function(w,x,N,Fitness,pop_size)
    M_interpolated = interpolation_function(x, x_vector, M_dist)
    return M_interpolated * fix_prob
@njit
def fixed_M_integrand(w, x, L_dist, x_vector,N,Fitness,pop_size, interpolation_function):
    fix_prob =  fixation_prob_function(w,x,N,Fitness,pop_size)
    L_interpolated = interpolation_function(w, x_vector, L_dist)
    return L_interpolated * fix_prob
def fixation_prob_integrand_del(x, L_dist, M_dist,x_vector,N,Fitness,pop_size, interpolation_function):
    M_interpolated = interpolation_function(x, x_vector, M_dist)
    integral_del, abs_error  = sp.quad(fixed_M_integrand, x, 1, args=(x, L_dist, x_vector,N,Fitness,pop_size, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    return M_interpolated * integral_del
def fixation_prob_integrand_ben(x, L_dist, M_dist,x_vector,N,Fitness,pop_size, interpolation_function):
    M_interpolated = interpolation_function(x, x_vector, M_dist)
    integral_ben, abs_error  = sp.quad(fixed_M_integrand, 0, x, args=(x, L_dist, x_vector,N,Fitness,pop_size, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    return M_interpolated * integral_ben
def compute_fixation_prob(L_dist, M_dist,x_vector,N,Fitness,pop_size, interpolation_function,highest_abs_error):
    result_ben, abs_error = sp.quad(fixation_prob_integrand_ben,0, 1, args=(L_dist,M_dist,x_vector,N,Fitness,pop_size,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # 0, 1
    result_del, abs_error = sp.quad(fixation_prob_integrand_del,0, 1, args=(L_dist,M_dist,x_vector,N,Fitness,pop_size,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # 0, 1
    highest_abs_error = max(abs_error,highest_abs_error)
    return result_ben, result_del, highest_abs_error
def fixed_L_distribution(L_dist, M_dist, fixation_prob,x_vector,N,Fitness,pop_size, interpolation_function, highest_abs_error):
    number_points = len(x_vector)
    fixed_L_dist = np.zeros(number_points)
    for i in range(number_points):
        integral, abs_error  = sp.quad(fixed_L_integrand, 0, 1, args=(x_vector[i], M_dist, x_vector, N,Fitness,pop_size,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)#integral, abs_error  = sp.quad(fixed_L_integrand, x_vector[i], 1, args=(x_vector[i], M_dist, x_vector, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        highest_abs_error = max(abs_error,highest_abs_error)
        fixed_L_dist[i] = (L_dist[i]/fixation_prob)*integral
    return fixed_L_dist, highest_abs_error
def fixed_M_distribution(L_dist, M_dist, fixation_prob,x_vector,N,Fitness,pop_size, interpolation_function, highest_abs_error):
    number_points = len(x_vector)
    fixed_M_dist = np.zeros(number_points)
    for i in range(number_points):
        integral, abs_error = sp.quad(fixed_M_integrand, 0, 1, args=(x_vector[i], L_dist, x_vector,N,Fitness,pop_size, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)#integral, abs_error = sp.quad(fixed_M_integrand, 0, x_vector[i], args=(x_vector[i], L_dist, x_vector, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        highest_abs_error = max(abs_error,highest_abs_error)
        fixed_M_dist[i] = (M_dist[i]/fixation_prob)*integral 
    return fixed_M_dist, highest_abs_error
@njit
def next_L_distribution(L_dist,fixed_L_dist,fixed_M_dist,x_vector,number_alleles):
    next_L_dist = np.zeros(len(L_dist))
    for i in range(len(L_dist)):
        next_L_dist[i] = L_dist[i] + fixed_M_dist[i]/number_alleles - fixed_L_dist[i]/number_alleles
    return next_L_dist
@njit
def next_M_distribution(M_dist, fixed_L_dist, fixed_M_dist,wVector,number_alleles):
    next_M_dist = np.zeros(len(M_dist))
    for i in range(len(M_dist)):
        next_M_dist[i] = M_dist[i] + fixed_L_dist[i]/number_alleles - fixed_M_dist[i]/number_alleles
    return next_M_dist
@njit
def chebishev_points(number_points,x_lim):
    points = np.zeros(number_points)
    for i in range(number_points):
        points[i] = x_lim[1]+x_lim[0]-(x_lim[1]+x_lim[0])/2-(x_lim[1]-x_lim[0])*np.cos(np.pi*(2*(i+1)-1)/(2*number_points))/2
    return points
def distributions_over_time(N, B, pop_size, number_mutations, x_vector, interpolation_method):
    number_points = len(x_vector)
    L_dist = np.ones((number_mutations+1,number_points))
    M_dist = np.ones((number_mutations+1,number_points))
    fixed_L_dist = np.ones((number_mutations+1,number_points))
    fixed_M_dist = np.ones((number_mutations+1,number_points))
    highest_abs_errors = np.zeros(number_mutations+1)
    Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(L_dist[0], x_vector, interpolation_method),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    fixation_prob_ben, fixation_prob_del, highest_abs_errors[0]  = compute_fixation_prob(L_dist[0], M_dist[0], x_vector,N,Fitness,pop_size, interpolation_method,highest_abs_errors[0])
    fixation_prob = fixation_prob_ben + fixation_prob_del
    fixed_L_dist[0], highest_abs_errors[0] = fixed_L_distribution(L_dist[0], M_dist[0], fixation_prob, x_vector,N,Fitness,pop_size, interpolation_method, highest_abs_errors[0])
    fixed_M_dist[0], highest_abs_errors[0] = fixed_M_distribution(L_dist[0], M_dist[0], fixation_prob, x_vector,N,Fitness,pop_size, interpolation_method, highest_abs_errors[0])
    for m in range(number_mutations):
        L_dist[m+1] = next_L_distribution(L_dist[m], fixed_L_dist[m], fixed_M_dist[m], x_vector, N)
        M_dist[m+1] = next_M_distribution(M_dist[m], fixed_L_dist[m], fixed_M_dist[m], x_vector, N*(B-1))
        Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(L_dist[m], x_vector, interpolation_method),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        fixation_prob_ben, fixation_prob_del, highest_abs_errors[m+1]   = compute_fixation_prob(L_dist[m+1], M_dist[m+1], x_vector,N,Fitness,pop_size, interpolation_method, highest_abs_errors[m+1] )
        fixation_prob = fixation_prob_ben + fixation_prob_del
        fixed_L_dist[m+1], highest_abs_errors[m+1] = fixed_L_distribution(L_dist[m+1], M_dist[m+1], fixation_prob, x_vector,N,Fitness,pop_size, interpolation_method, highest_abs_errors[m+1])
        fixed_M_dist[m+1], highest_abs_errors[m+1] = fixed_M_distribution(L_dist[m+1], M_dist[m+1], fixation_prob, x_vector,N,Fitness,pop_size, interpolation_method, highest_abs_errors[m+1])
    #print("Highest abs global errors for integration per iteration ",highest_abs_errors/number_points)
    return L_dist, M_dist, fixed_L_dist, fixed_M_dist 
@njit
def integrand_dfe(w,s,L_dist,M_dist,x_vector,N,W, interpolation_function):
    L_interpolated = interpolation_function(w, x_vector, L_dist)
    x = N*s*W+w
    M_interpolated = interpolation_function(x, x_vector, M_dist)
    return N*L_interpolated*M_interpolated*W

def computeDFE(Ldist,Mdist,x_vector,N,W,interpolation_function):
    s_range = np.array([-1/(N*W),1/(N*W)])
    s_points = chebishev_points(len(x_vector),s_range)
    DFE = np.zeros(len(s_points))
    DFE_higher_abs_error = 0
    i = 0
    for s in s_points:
        if s <=0:
            lowBound = -N*s*W
            val = sp.quad(integrand_dfe,lowBound,1,args = (s,Ldist,Mdist,x_vector,N,W,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # lowBound, 1
        elif s > 0:
            uppBound = 1 - N*s*W
            val = sp.quad(integrand_dfe,0,uppBound,args = (s,Ldist,Mdist,x_vector,N,W,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # 0, uppBound
        DFE[i]+=val[0]      
        if val[1] > DFE_higher_abs_error:
            DFE_higher_abs_error = val[1]
            #print("New higher abs error =",DFE_higher_abs_error)
        i+=1
    return s_points, DFE

@njit
def distributions_mean_integrand(w,distribution,x_vector,interpolation_function):
    L_interpolated = interpolation_function(w,x_vector,distribution)
    return L_interpolated*w

@njit
def distributions_var_integrand(w,distribution,x_vector,mean,interpolation_function):
    L_interpolated = interpolation_function(w,x_vector,distribution)
    return L_interpolated*(mean - w)**2


def distributions_statistics(Ldist,Mdist,LFdist,MFdist,x_vector,interpolation_function):
    L_mean, error = sp.quad(distributions_mean_integrand, 0, 1, args=(Ldist, x_vector, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    M_mean, error = sp.quad(distributions_mean_integrand, 0, 1, args=(Mdist, x_vector, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    LF_mean, error = sp.quad(distributions_mean_integrand, 0, 1, args=(LFdist, x_vector, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    MF_mean, error = sp.quad(distributions_mean_integrand, 0, 1, args=(MFdist, x_vector, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    
    L_var, error = sp.quad(distributions_var_integrand, 0, 1, args=(Ldist, x_vector,L_mean, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    M_var, error = sp.quad(distributions_var_integrand, 0, 1, args=(Mdist, x_vector,M_mean, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    LF_var, error = sp.quad(distributions_var_integrand, 0, 1, args=(LFdist, x_vector,LF_mean, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    MF_var, error = sp.quad(distributions_var_integrand, 0, 1, args=(MFdist, x_vector,MF_mean, interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    return L_mean, M_mean, LF_mean, MF_mean, L_var, M_var, LF_var, MF_var



def DFE_mean_integrand(s,splines_DFE):
    return s*splines_DFE(s)



def DFE_statistics(Ldist,Mdist,w_points,N,interpolation_method):
    Fitness, error = sp.quad(distributions_mean_integrand, 0, 1, args=(Ldist, w_points, interpolation_method),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    s_points, DFE_ana = computeDFE(Ldist,Mdist,w_points,N,Fitness,interpolation_method)
    splines_DFE = si.CubicSpline(s_points, DFE_ana)
    beneficial_fraction, error_ben = sp.quad(splines_DFE,0, s_points[-1], args=(),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    deleterious_fraction, error_del = sp.quad(splines_DFE, s_points[0], 0, args=(),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    DFE_mean, error = sp.quad(DFE_mean_integrand,s_points[0], s_points[-1], args=(splines_DFE),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    DFE_ben_mean, error = sp.quad(DFE_mean_integrand,0, s_points[-1], args=(splines_DFE),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    DFE_del_mean, error = sp.quad(DFE_mean_integrand,s_points[0], 0, args=(splines_DFE),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    return DFE_mean, DFE_ben_mean, DFE_del_mean, beneficial_fraction, deleterious_fraction




def computeAllDFE(Ldist,Mdist,x_vector,N,interpolation_function):
    s_points = np.zeros(Ldist.shape)
    DFE = np.zeros(Ldist.shape)
    DFE_higher_abs_error = 0
    for mut in range(len(Ldist)):
        Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(Ldist[mut], x_vector, chebishev_interpolation),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        s_range = np.array([-1/(N*Fitness),1/(N*Fitness)])
        s_points[mut] = chebishev_points(len(x_vector),s_range)
        i = 0
        for s in s_points[mut]:
            if s <=0:
                lowBound = -N*s*Fitness
                val = sp.quad(integrand_dfe,lowBound,1,args = (s,Ldist[mut],Mdist[mut],x_vector,N,Fitness,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # lowBound, 1
            elif s > 0:
                uppBound = 1 - N*s*Fitness
                val = sp.quad(integrand_dfe,0,uppBound,args = (s,Ldist[mut],Mdist[mut],x_vector,N,Fitness,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # 0, uppBound
            DFE[mut,i]+=val[0]      
            if val[1] > DFE_higher_abs_error:
                DFE_higher_abs_error = val[1]
                #print("New higher abs error =",DFE_higher_abs_error)
            i+=1
    return s_points, DFE



def computeAllDFE_ben(Ldist,Mdist,x_vector,N,interpolation_function):
    s_points = np.zeros(Ldist.shape)
    DFE = np.zeros(Ldist.shape)
    DFE_higher_abs_error = 0
    for mut in range(len(Ldist)):
        Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(Ldist[mut], x_vector, chebishev_interpolation),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        s_range = np.array([0,1/(N*Fitness)])
        s_points[mut] = chebishev_points(len(x_vector),s_range)
        i = 0
        for s in s_points[mut]:
            if s <0:
                lowBound = -N*s*Fitness
                val = sp.quad(integrand_dfe,lowBound,1,args = (s,Ldist[mut],Mdist[mut],x_vector,N,Fitness,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # lowBound, 1
            elif s >= 0:
                uppBound = 1 - N*s*Fitness
                val = sp.quad(integrand_dfe,0,uppBound,args = (s,Ldist[mut],Mdist[mut],x_vector,N,Fitness,interpolation_function),epsabs=10**(-8), epsrel=10**(-8), limit=10**6) # 0, uppBound
            DFE[mut,i]+=val[0]      
            if val[1] > DFE_higher_abs_error:
                DFE_higher_abs_error = val[1]
                #print("New higher abs error =",DFE_higher_abs_error)
            i+=1
        splines_DFE = si.CubicSpline(s_points[mut], DFE[mut])
        beneficial_fraction, error_ben = sp.quad(splines_DFE,s_points[mut,0], s_points[mut,-1], args=(),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        DFE[mut] = DFE[mut]/beneficial_fraction
    return s_points, DFE


def computeAllDFE_plot(Ldist,Mdist,x_vector,N,interpolation_function):
    number_points_for_DFE = 100
    s_points = np.zeros((len(Ldist),number_points_for_DFE))
    DFE = np.zeros((len(Ldist),number_points_for_DFE))
    DFE_higher_abs_error = 0
    for mut in range(len(Ldist)):
        Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(Ldist[mut], x_vector, chebishev_interpolation),epsabs=10**(-4), epsrel=10**(-4), limit=10**6)
        s_points[mut] = np.linspace(-1/(N*Fitness),1/(N*Fitness),number_points_for_DFE)
        i = 0
        for s in s_points[mut]:
            if s <0:
                lowBound = -N*s*Fitness
                val = sp.quad(integrand_dfe,lowBound,1,args = (s,Ldist[mut],Mdist[mut],x_vector,N,Fitness,interpolation_function),epsabs=10**(-4), epsrel=10**(-4), limit=10**6) # lowBound, 1
            elif s > 0:
                uppBound = 1 - N*s*Fitness
                val = sp.quad(integrand_dfe,0,uppBound,args = (s,Ldist[mut],Mdist[mut],x_vector,N,Fitness,interpolation_function),epsabs=10**(-4), epsrel=10**(-4), limit=10**6) # 0, uppBound
            DFE[mut,i]+=val[0]      
            if val[1] > DFE_higher_abs_error:
                DFE_higher_abs_error = val[1]
                #print("New higher abs error =",DFE_higher_abs_error)
            i+=1
    return s_points, DFE



@njit
def beneficial_DFE_for_data(data_DFE):
    number_of_ben_fe = 0
    for i in range(len(data_DFE)):
        if data_DFE[i] >= 0:
            number_of_ben_fe += 1
    data_DFE_ben = np.zeros(number_of_ben_fe)
    count = 0
    for j in range(len(data_DFE)):
        if data_DFE[j] >= 0:
            data_DFE_ben[count] = data_DFE[j]
            count += 1
    return data_DFE_ben



































## SET INITIAL CONDITIONS 
N = 100
B = 4
m = 200
pop_size = 1000
file_for_reading = 'data_additive_SI'
## RUN ANALYTICAL FIRST BECAUSE ITS FASTER
number_points = 30     ## ~20 BEHAVES PROPERLY, DO NOT GET HIGHER THAN 30
number_points_for_detailed = 100
w_range = np.array([0,1])
w_points = chebishev_points(number_points,w_range)
w_points_detailed = np.linspace(w_points[0], w_points[-1],number_points_for_detailed)
L_ana,M_ana,LF_ana,MF_ana = distributions_over_time(N,B,pop_size,m,w_points,chebishev_interpolation)
Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(L_ana[-1], w_points, chebishev_interpolation),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
fix_prob_ben,fix_prob_del, error = compute_fixation_prob(L_ana[-1], M_ana[-1],w_points,N,Fitness,pop_size, chebishev_interpolation,0)
print('Beneficial fixation probability  =',fix_prob_ben,'\nDeleterious fixation probability =',fix_prob_del)
print('Probability of a ben. substitution',fix_prob_ben/(fix_prob_ben+fix_prob_del))
number_points_plot = 100
plot_points = np.linspace(0,1,number_points_plot)
distributions_plot = np.zeros((4,number_points_plot))
for i in range(number_points_plot):
    distributions_plot[0][i] = chebishev_interpolation(plot_points[i],w_points,L_ana[-1])
    distributions_plot[1][i] = chebishev_interpolation(plot_points[i],w_points,M_ana[-1])
    distributions_plot[2][i] = chebishev_interpolation(plot_points[i],w_points,LF_ana[-1])
    distributions_plot[3][i] = chebishev_interpolation(plot_points[i],w_points,MF_ana[-1])
plt.figure(dpi=150)
plt.plot(plot_points,distributions_plot[0],'-',color=[0,0,1])
plt.plot(plot_points,distributions_plot[1],'-',color=[1,0,0])
plt.plot(plot_points,distributions_plot[2],'-',color=[0,1,1])
plt.plot(plot_points,distributions_plot[3],'--',color=[1,0,1])
plt.xlabel('Fitness contribution $(w,x)$')
plt.ylabel('Probability density')
plt.legend(['$L(w)$','$M(x)$','$L_F(w)$','$M_F(x)$'])
plt.grid(True)
## COMPUTE THE DFE
Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(L_ana[-1], w_points, chebishev_interpolation),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
s_points, DFE_ana = computeDFE(L_ana[-1],M_ana[-1],w_points,N,Fitness,chebishev_interpolation)
splines_DFE = si.CubicSpline(s_points, DFE_ana)
s_plot = np.zeros((2,number_points_plot))
s_plot[-1] = np.linspace(s_points[0],s_points[-1],number_points_plot)
DFE_plot = np.zeros((2,len(s_plot[0])))
DFE_plot[-1] = splines_DFE(s_plot[-1])

Fitness,error = sp.quad(distributions_mean_integrand, 0, 1, args=(L_ana[0], w_points, chebishev_interpolation),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
s_points, DFE_ana = computeDFE(L_ana[0],M_ana[0],w_points,N,Fitness,chebishev_interpolation)
s_plot[0] = np.linspace(s_points[0],s_points[-1],number_points_plot)
splines_DFE = si.CubicSpline(s_points, DFE_ana)
DFE_plot[0] = splines_DFE(s_plot[0])



L_mean, M_mean, LF_mean, MF_mean, L_var, M_var, LF_var, MF_var = distributions_statistics(L_ana[-1],M_ana[-1],LF_ana[-1],MF_ana[-1],w_points,chebishev_interpolation)
print("\nMeans for the distributions",L_mean, M_mean, LF_mean, MF_mean)
print("Standard deviation for the distributions",L_var**0.5, M_var**0.5, LF_var**0.5, MF_var**0.5)


DFE_mean, DFE_ben_mean, DFE_del_mean, beneficial_fraction, deleterious_fraction = DFE_statistics(L_ana[-1],M_ana[-1],w_points,N,chebishev_interpolation)
print("\nBeneficial fraction = ",beneficial_fraction,'Deleterious fraction = ',deleterious_fraction,'Error =',abs(1-beneficial_fraction-deleterious_fraction))
print("overall mean, ben mean and del mean",DFE_mean, DFE_ben_mean, DFE_del_mean)





mut_points = np.arange(m+1)
distributions_mean = np.zeros((4,m+1))
distributions_sd = np.zeros((4,m+1))
DFE_stats = np.zeros((3,m+1))
for i in range(m+1):
    L_mean,M_mean,LF_mean,MF_mean,L_sd,M_sd,LF_sd,MF_sd = distributions_statistics(L_ana[mut_points[i]],M_ana[mut_points[i]],LF_ana[mut_points[i]],MF_ana[mut_points[i]],w_points,chebishev_interpolation)
    distributions_mean[0][i] = L_mean
    distributions_mean[1][i] = M_mean
    distributions_mean[2][i] = LF_mean
    distributions_mean[3][i] = MF_mean
    distributions_sd[0][i] = np.sqrt(L_sd)
    distributions_sd[1][i]= np.sqrt(M_sd)
    distributions_sd[2][i]= np.sqrt(LF_sd)
    distributions_sd[3][i]= np.sqrt(MF_sd)
    DFE_mean, DFE_ben_mean, DFE_del_mean, beneficial_fraction, deleterious_fraction = DFE_statistics(L_ana[mut_points[i]],M_ana[mut_points[i]],w_points,N,chebishev_interpolation)
    DFE_stats[0][i] = DFE_ben_mean
    DFE_stats[1][i] = DFE_del_mean
    DFE_stats[2][i] = beneficial_fraction



data_L_boxplots = np.loadtxt(file_for_reading+"\Ldistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
data_M_boxplots = np.loadtxt(file_for_reading+"\Mdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
data_LF_boxplots = np.loadtxt(file_for_reading+"\LFdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
data_MF_boxplots = np.loadtxt(file_for_reading+"\MFdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")

number_boxplots = 11
mut_boxplots = np.arange(number_boxplots)
mut_boxplots = mut_boxplots*int(m/(number_boxplots-1))
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
plt.plot(mut_points,distributions_mean[0],'--',color=[0,0,1])
plt.plot(mut_points,distributions_mean[1],'--',color=[1,0,0])
plt.fill_between(mut_points, distributions_mean[0]-distributions_sd[0],distributions_mean[0]+distributions_sd[0], alpha=0.4,color=[0,0,1])
plt.fill_between(mut_points, distributions_mean[1]-distributions_sd[1],distributions_mean[1]+distributions_sd[1], alpha=0.4,color=[1,0,0])
plt.boxplot(data_L_boxplots,positions=mut_boxplots,notch=True,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_L,meanprops=meanprops_L,whiskerprops=whiskerprops_L)
plt.boxplot(data_M_boxplots,positions=mut_boxplots+m/200,notch=True,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_M,meanprops=meanprops_M,whiskerprops=whiskerprops_M)
plt.xticks(mut_boxplots)
plt.legend(['$L(w)$','$M(x)$'],loc='lower right')
plt.xlabel('Substitutions')
plt.ylabel('Fitness contribution')


plt.figure(dpi=150)
plt.plot(mut_points,distributions_mean[2],'--',color=[0,1,1])
plt.plot(mut_points,distributions_mean[3],'--',color=[1,0,1])
plt.fill_between(mut_points, distributions_mean[2]-distributions_sd[2],distributions_mean[2]+distributions_sd[2], alpha=0.4,color=[0,1,1])
plt.fill_between(mut_points, distributions_mean[3]-distributions_sd[3],distributions_mean[3]+distributions_sd[3], alpha=0.4,color=[1,0,1])
plt.boxplot(data_LF_boxplots,positions=mut_boxplots,notch=True,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_LF,meanprops=meanprops_LF,whiskerprops=whiskerprops_LF)
plt.boxplot(data_MF_boxplots,positions=mut_boxplots+m/200,notch=True,widths=m/20,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_MF,meanprops=meanprops_MF,whiskerprops=whiskerprops_MF)
plt.xticks(mut_boxplots)
plt.legend(['$L_F(w)$','$M_F(x)$'],loc='lower right')
plt.xlabel('Substitutions')
plt.ylabel('Fitness contribution')



plt.figure(dpi=150)
plt.plot(plot_points,distributions_plot[0],'-',color=[0,0,1])
plt.plot(plot_points,distributions_plot[1],'-',color=[1,0,0])
plt.plot(plot_points,distributions_plot[2],'-',color=[0,1,1])
plt.plot(plot_points,distributions_plot[3],'--',color=[1,0,1])
plt.hist(data_L_boxplots[:,-1],alpha=0.4,density=True,bins=100,color=[0,0,1])
plt.hist(data_M_boxplots[:,-1],alpha=0.4,density=True,bins=100,color=[1,0,0])
plt.hist(data_LF_boxplots[:,-1],alpha=0.4,density=True,bins=100,color=[0,1,1])
plt.hist(data_MF_boxplots[:,-1],alpha=0.4,density=True,bins=100,color=[1,0,1])
plt.title('G ='+str(N)+', B ='+str(B)+',N ='+str(pop_size))
plt.xlabel('Fitness contribution $(w,x)$')
plt.ylabel('Probability density')
plt.legend(['$L(w)$','$M(x)$','$L_F(w)$','$M_F(x)$'])
plt.grid(True)



# data_DFE_boxplots = np.loadtxt("data_additive_104\DFEdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
# plt.figure(dpi=150)
# plt.plot(s_plot[0],DFE_plot[0],'-',color=[0,0.8,0])
# plt.hist(data_DFE_boxplots[:,0],alpha=0.4,density=True,bins=100,color=[0,0.8,0])
# plt.plot(s_plot[-1],DFE_plot[-1],'-',color=[0,0.8,0])
# plt.hist(data_DFE_boxplots[:,-1],alpha=0.4,density=True,bins=100,color=[0,0.8,0])
# plt.grid(True)
# plt.legend(['DFE'],loc='upper right')
# plt.title('N ='+str(N)+', B ='+str(B)+', P ='+str(pop_size))
# plt.xlabel('Fitness effect')
# plt.ylabel('Probability density')





## THIS CHUNK COMPUTES THE CUMULATIVE DISTRIBUTIONS
L_cumulative = np.zeros(L_ana.shape)
M_cumulative = np.zeros(M_ana.shape)
LF_cumulative = np.zeros(LF_ana.shape)
MF_cumulative = np.zeros(MF_ana.shape)
DFE_cumulative = np.zeros(DFE_ana.shape)

for i in range(m+1):
    for j in range(number_points):
        L_cumulative[i,j], error = sp.quad(chebishev_interpolation, 0, w_points[j], args=(w_points, L_ana[i]),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        M_cumulative[i,j], error = sp.quad(chebishev_interpolation, 0, w_points[j], args=(w_points, M_ana[i]),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        LF_cumulative[i,j], error = sp.quad(chebishev_interpolation, 0, w_points[j], args=(w_points, LF_ana[i]),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
        MF_cumulative[i,j], error = sp.quad(chebishev_interpolation, 0, w_points[j], args=(w_points, MF_ana[i]),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
## END CHUNK 

## THIS CHUNK PLOT THE CUMULATIVE DISTRIBUTIONS
plt.figure(dpi=150)
plt.plot(w_points,L_cumulative[0])
plt.plot(w_points,L_cumulative[-1])
plt.plot(w_points,M_cumulative[0])
plt.plot(w_points,M_cumulative[-1])
plt.plot(w_points,LF_cumulative[0])
plt.plot(w_points,LF_cumulative[-1])
plt.plot(w_points,MF_cumulative[0])
plt.plot(w_points,MF_cumulative[-1])
## END CHUNK 



## THIS CHUNK COMPUTES THE QUARTILES FOR THE DISTRIBUTIONS
quartiles_percentages = np.array([0.025,0.25,0.5,0.75,0.975])
steps_for_quartiles = 10**3
L_anal_quartiles = np.zeros((m+1,len(quartiles_percentages)))
M_anal_quartiles = np.zeros((m+1,len(quartiles_percentages)))
LF_anal_quartiles = np.zeros((m+1,len(quartiles_percentages)))
MF_anal_quartiles = np.zeros((m+1,len(quartiles_percentages)))
for q in range(len(quartiles_percentages)):
    for mut in range(m+1):
        for w in np.linspace(0,1,steps_for_quartiles):
            L_cum_w = chebishev_interpolation(w,w_points,L_cumulative[mut])
            if L_cum_w >= quartiles_percentages[q]:
                L_anal_quartiles[mut,q] = w
                break
        for w in np.linspace(0,1,steps_for_quartiles):
            M_cum_w = chebishev_interpolation(w,w_points,M_cumulative[mut])
            if M_cum_w >= quartiles_percentages[q]:
                M_anal_quartiles[mut,q] = w
                break
        for w in np.linspace(0,1,steps_for_quartiles):
            LF_cum_w = chebishev_interpolation(w,w_points,LF_cumulative[mut])
            if LF_cum_w >= quartiles_percentages[q]:
                LF_anal_quartiles[mut,q] = w
                break
        for w in np.linspace(0,1,steps_for_quartiles):
            MF_cum_w = chebishev_interpolation(w,w_points,MF_cumulative[mut])
            if MF_cum_w >= quartiles_percentages[q]:
                MF_anal_quartiles[mut,q] = w
                break
## END CHUNK

## PLOT THE BOXPLOTS AND QUARTILE AREAS
number_boxplots = 11
mut_boxplots = np.arange(number_boxplots)
mut_boxplots = mut_boxplots*int(m/(number_boxplots-1))
boxprops_L = dict(linestyle='-', linewidth=1, color=[0,0,0.8])
meanprops_L =dict(linestyle='--', linewidth=1, color=[0,0,0.7])
whiskerprops_L =dict(linestyle='-', linewidth=1, color=[0,0,0.8])
medianprops_L = dict(linewidth=0)
boxprops_M = dict(linestyle='-', linewidth=1, color=[0.8,0,0])
meanprops_M =dict(linestyle='--', linewidth=1, color=[0.7,0,0])
whiskerprops_M =dict(linestyle='-', linewidth=1, color=[0.8,0,0])
medianprops_M = dict(linewidth=0)
boxprops_LF = dict(linestyle='-', linewidth=1, color=[0,0.8,0.8])
meanprops_LF =dict(linestyle='--', linewidth=1, color=[0,0.7,0.7])
whiskerprops_LF =dict(linestyle='-', linewidth=1, color=[0,0.8,0.8])
medianprops_LF = dict(linewidth=0)
boxprops_MF = dict(linestyle='-', linewidth=1, color=[0.8,0,0.8])
medianprops_MF =dict(linewidth=0)
whiskerprops_MF =dict(linestyle='-', linewidth=1, color=[0.8,0,0.8])
meanprops_MF = dict(linestyle='--', linewidth=1, color=[0.7,0,0.7])

plt.figure(dpi=150)
plt.plot(mut_points,distributions_mean[0],'--',color=[0,0,0.7])
plt.plot(mut_points,distributions_mean[1],'--',color=[0.7,0,0])
plt.fill_between(mut_points, L_anal_quartiles[:,1],L_anal_quartiles[:,3], alpha=0.4,color=[0,0,1])
plt.fill_between(mut_points, M_anal_quartiles[:,1],M_anal_quartiles[:,3], alpha=0.4,color=[1,0,0])
plt.fill_between(mut_points, L_anal_quartiles[:,0],L_anal_quartiles[:,4], alpha=0.2,color=[0,0,1])
plt.fill_between(mut_points, M_anal_quartiles[:,0],M_anal_quartiles[:,4], alpha=0.2,color=[1,0,0])
L_box = plt.boxplot(data_L_boxplots,positions=mut_boxplots,whis=(2.5, 97.5),widths=m/20,medianprops=medianprops_L,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_L,meanprops=meanprops_L,whiskerprops=whiskerprops_L)
M_box = plt.boxplot(data_M_boxplots,positions=mut_boxplots+m/200,whis=(2.5, 97.5),widths=m/20,medianprops=medianprops_M,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_M,meanprops=meanprops_M,whiskerprops=whiskerprops_M)
plt.setp(M_box["caps"], color=[0.8,0,0])
plt.setp(L_box["caps"], color=[0,0,0.8])
plt.xticks(mut_boxplots)
plt.legend(['$L(w)$','$M(x)$'],loc='lower right')
plt.xlabel('Substitutions')
plt.ylabel('Fitness contribution')
plt.figure(dpi=150)
plt.plot(mut_points,distributions_mean[2],'--',color=[0,0.7,0.7])
plt.plot(mut_points,distributions_mean[3],'--',color=[0.7,0,0.7])
plt.fill_between(mut_points, LF_anal_quartiles[:,1],LF_anal_quartiles[:,3], alpha=0.4,color=[0,1,1])
plt.fill_between(mut_points, MF_anal_quartiles[:,1],MF_anal_quartiles[:,3], alpha=0.4,color=[1,0,1])
plt.fill_between(mut_points, LF_anal_quartiles[:,0],LF_anal_quartiles[:,4], alpha=0.2,color=[0,1,1])
plt.fill_between(mut_points, MF_anal_quartiles[:,0],MF_anal_quartiles[:,4], alpha=0.2,color=[1,0,1])
LF_box=plt.boxplot(data_LF_boxplots,positions=mut_boxplots,whis=(2.5, 97.5),widths=m/20,medianprops=medianprops_LF,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_LF,meanprops=meanprops_LF,whiskerprops=whiskerprops_LF)
MF_box=plt.boxplot(data_MF_boxplots,positions=mut_boxplots+m/200,whis=(2.5, 97.5),widths=m/20,medianprops=medianprops_MF,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_MF,meanprops=meanprops_MF,whiskerprops=whiskerprops_MF)
plt.setp(MF_box["caps"], color=[0.8,0,0.8])
plt.setp(LF_box["caps"], color=[0,0.8,0.8])
plt.xticks(mut_boxplots)
plt.legend(['$L_F(w)$','$M_F(x)$'],loc='lower right')
plt.xlabel('Substitutions')
plt.ylabel('Fitness contribution')
## END CHUNK



s_points, DFE_ana = computeAllDFE(L_ana, M_ana, w_points, N, chebishev_interpolation)

DFE_cumulative = np.zeros(DFE_ana.shape)
for mut in range(m+1):
    splines_DFE = si.CubicSpline(s_points[mut], DFE_ana[mut])
    for j in range(number_points):
        DFE_cumulative[mut,j], error = sp.quad(splines_DFE, s_points[mut,0], s_points[mut,j], args=(),epsabs=10**(-4), epsrel=10**(-4), limit=10**6)


DFE_anal_quartiles = np.zeros((m+1,len(quartiles_percentages)))
for mut in range(m+1):
    for q in range(len(quartiles_percentages)):
        for s in np.linspace(s_points[mut,0],s_points[mut,-1],steps_for_quartiles):
            DFE_cum_s = chebishev_interpolation(s,s_points[mut],DFE_cumulative[mut])
            if DFE_cum_s >= quartiles_percentages[q]:
                DFE_anal_quartiles[mut,q] = s
                break
            
data_DFE_boxplots = np.loadtxt(file_for_reading+"\DFEdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")

DFE_samples_per_walk = N*(B-1)
total_walks = int(len(data_DFE_boxplots)/DFE_samples_per_walk)
data_DFE_ben_frac = np.zeros((total_walks,len(data_DFE_boxplots[0])))

for mut in range(len(data_DFE_boxplots[0])):
    walk = 0
    while walk < total_walks:
        for i in range(DFE_samples_per_walk):
            s = data_DFE_boxplots[walk*DFE_samples_per_walk + i,mut]
            if s > 0:
                data_DFE_ben_frac[walk,mut] += 1 
        walk+=1
data_DFE_ben_frac = data_DFE_ben_frac/DFE_samples_per_walk



plt.figure(dpi=150)
plt.xlabel("Substitutions")    
plt.fill_between(mut_points, DFE_anal_quartiles[:,1],DFE_anal_quartiles[:,3], alpha=0.4,color=[0,1,0])
plt.fill_between(mut_points, DFE_anal_quartiles[:,0],DFE_anal_quartiles[:,4], alpha=0.2,color=[0,1,0])
plt.plot(mut_points,DFE_stats[0]+DFE_stats[1],'--',color=[0,0.7,0])


boxprops_DFE = dict(linestyle='-', linewidth=1, color=[0,0.8,0])
meanprops_DFE =dict(linestyle='--', linewidth=1, color=[0,0.7,0])
whiskerprops_DFE =dict(linestyle='-', linewidth=1, color=[0,0.8,0])
medianprops_DFE = dict(linewidth=0)
DFE_box=plt.boxplot(data_DFE_boxplots,positions=mut_boxplots,whis=(2.5, 97.5),widths=m/20,medianprops=medianprops_DFE,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_DFE,meanprops=meanprops_DFE,whiskerprops=whiskerprops_DFE)
plt.setp(DFE_box["caps"], color=[0,0.8,0])
plt.ylabel('Fitness effect (s)',color=[0,1,0])
plt.tick_params(labelcolor=[0,1,0])
ax2 = plt.twinx() 
ax2.plot(mut_points,DFE_stats[2],'-',color=[0.8,0.8,0])   
plt.ylabel('Beneficial fraction',color=[0.8,0.8,0])
plt.tick_params(labelcolor=[0.8,0.8,0])
boxprops_ben_frac = dict(linestyle='-', linewidth=1, color=[0.8,0.8,0])
meanprops_ben_frac =dict(linestyle='--', linewidth=0, color=[0.8,0.8,0])
whiskerprops_ben_frac =dict(linestyle='-', linewidth=1, color=[0.8,0.8,0])
medianprops_ben_frac = dict(linestyle='-', linewidth=1, color=[0.8,0.8,0])
DFE_box_ben_frac=plt.boxplot(data_DFE_ben_frac,positions=mut_boxplots+m/200,whis=(2.5, 97.5),widths=m/20,medianprops=medianprops_ben_frac,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_ben_frac,meanprops=meanprops_ben_frac,whiskerprops=whiskerprops_ben_frac)
plt.setp(DFE_box_ben_frac["caps"], color=[0.8,0.8,0])
plt.xticks(mut_boxplots)





#Distributions histograms and lines at three points

fig, axs = plt.subplots(2,2,sharex=True,dpi=150, layout='constrained')
axs[1,1].plot([],[],color=(0.8,0.8,0.8))
axs[1,1].plot([],[],color=(0.5,0.5,0.5))
axs[1,1].plot([],[],color=(0.1,0.1,0.1))
axs[0,0].set_ylabel("Probability density, $L_i$")
axs[0,1].set_ylabel("Probability density, $M_i$")
axs[1,0].set_xlabel('Fitness contribution, $w$')
axs[1,0].set_ylabel("Probability density, $L_{i|F}$")
axs[1,1].set_xlabel("Fitness contribution, $w'$ ")
axs[1,1].set_ylabel("Probability density, $M_{i|F}$")
axs[1,1].legend(('i = '+str(mut_boxplots[0]),'i = '+str(mut_boxplots[3]),'i = '+str(mut_boxplots[-1])))
axs[0,0].hist(data_L_boxplots[:,0],density=True,bins=100,alpha=0.4,color=((0,0,1)))
axs[0,0].hist(data_L_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0,0,0.6))
axs[0,0].hist(data_L_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0,0,0.3))
#For L
L_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    L_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, L_ana[0])
axs[0,0].plot(w_points_detailed,L_ana_detailed,color=(0,0,1))
L_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    L_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, L_ana[60]) 
axs[0,0].plot(w_points_detailed,L_ana_detailed,color=(0,0,0.6))
L_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    L_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, L_ana[-1]) 
axs[0,0].plot(w_points_detailed,L_ana_detailed,color=(0,0,0.3))
# For LF
LF_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    LF_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, LF_ana[0])
axs[1,0].plot(w_points_detailed,LF_ana_detailed,color=(0,0.9,0.9))
LF_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    LF_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, LF_ana[60])
axs[1,0].plot(w_points_detailed,LF_ana_detailed,color=(0,0.6,0.6))
LF_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    LF_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, LF_ana[-1])
axs[1,0].plot(w_points_detailed,LF_ana_detailed,color=(0,0.3,0.3))
axs[1,0].hist(data_LF_boxplots[:,0],density=True,bins=100,alpha=0.4,color=(0,0.9,0.9))
axs[1,0].hist(data_LF_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0,0.6,0.6))
axs[1,0].hist(data_LF_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0,0.3,0.3))
# For M
M_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    M_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, M_ana[0])
axs[0,1].plot(w_points_detailed,M_ana_detailed,color=(1,0,0))
M_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    M_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, M_ana[60])
axs[0,1].plot(w_points_detailed,M_ana_detailed,color=(0.6,0,0))
M_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    M_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, M_ana[-1])
axs[0,1].plot(w_points_detailed,M_ana_detailed,color=(0.3,0,0))
axs[0,1].hist(data_M_boxplots[:,0],density=True,bins=100,alpha=0.4,color=(1,0,0))
axs[0,1].hist(data_M_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0.6,0,0))
axs[0,1].hist(data_M_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0.3,0,0))
#For MF
MF_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    MF_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, MF_ana[0])
axs[1,1].plot(w_points_detailed,MF_ana_detailed,color=(1,0,1))
MF_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    MF_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, MF_ana[60])
axs[1,1].plot(w_points_detailed,MF_ana_detailed,color=(0.6,0,0.6))
MF_ana_detailed = np.zeros(number_points_for_detailed)
for i in range(number_points_for_detailed):
    MF_ana_detailed[i] = chebishev_interpolation(w_points_detailed[i], w_points, MF_ana[-1])
axs[1,1].plot(w_points_detailed,MF_ana_detailed,color=(0.3,0,0.3))
axs[1,1].hist(data_MF_boxplots[:,0],density=True,bins=100,alpha=0.5,color=(1,0,1))
axs[1,1].hist(data_MF_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0.6,0,0.6))
axs[1,1].hist(data_MF_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0.3,0,0.3))
axs[1,0].set_ylim([0,max(MF_ana[:,-1])])
axs[1,1].set_ylim([0,max(MF_ana[:,-1])])

# fig, axs = plt.subplots(1,2,sharex=True,dpi=150, layout='constrained')
# axs[0].plot(w_points,L_ana[0],color=(0,0,1))
# #axs[0].plot(w_points,L_ana[60],color=(0,0,0.6))
# axs[0].plot(w_points,L_ana[-1],color=(0,0,0.6))
# axs[0].hist(data_L_boxplots[:,0],density=True,bins=100,alpha=0.4,color=((0,0,1)))
# #axs[0].hist(data_L_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0,0,0.6))
# axs[0].hist(data_L_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0,0,0.6))

# axs[1].plot(w_points,LF_ana[0],color=(0,1,1))
# #axs[1].plot(w_points,LF_ana[60],color=(0,0.6,0.6))
# axs[1].plot(w_points,LF_ana[-1],color=(0,0.6,0.6))
# axs[1].hist(data_LF_boxplots[:,0],density=True,bins=100,alpha=0.4,color=(0,1,1))
# #axs[1].hist(data_LF_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0,0.6,0.6))
# axs[1].hist(data_LF_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0,0.6,0.6))

# axs[0].plot(w_points,M_ana[0],color=(1,0,0))
# #axs[0].plot(w_points,M_ana[60],color=(0.6,0,0))
# axs[0].plot(w_points,M_ana[-1],color=(0.6,0,0))
# axs[0].hist(data_M_boxplots[:,0],density=True,bins=100,alpha=0.4,color=(1,0,0))
# #axs[0].hist(data_M_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0.6,0,0))
# axs[0].hist(data_M_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0.6,0,0))

# axs[1].plot(w_points,MF_ana[0],color=(1,0,1))
# #axs[1].plot(w_points,MF_ana[60],color=(0.6,0,0.6))
# axs[1].plot(w_points,MF_ana[-1],color=(0.6,0,0.6))
# axs[1].hist(data_MF_boxplots[:,0],density=True,bins=100,alpha=0.4,color=(1,0,1))
# #axs[1].hist(data_MF_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0.6,0,0.6))
# axs[1].hist(data_MF_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0.6,0,0.6))





#DFE HISTOGRAMS OVER TIME
s_points_plot, DFE_ana_plot = computeAllDFE_plot(L_ana, M_ana, w_points, N, chebishev_interpolation)
plt.figure(dpi=150)
plt.plot([],[],color=(0.8,0.8,0.8))
plt.plot([],[],color=(0.5,0.5,0.5))
plt.plot([],[],color=(0.1,0.1,0.1))
plt.ylabel("Probability density")
plt.xlabel('Fitness effect, $s$')
plt.legend(('i = '+str(mut_boxplots[0]),'i = '+str(mut_boxplots[3]),'i = '+str(mut_boxplots[-1])))
plt.plot(s_points_plot[0], DFE_ana_plot[0],color=(0,0.9,0))
plt.plot(s_points_plot[60], DFE_ana_plot[60],color=(0,0.5,0))
plt.plot(s_points_plot[-1], DFE_ana_plot[-1],color=(0,0.3,0))

plt.hist(data_DFE_boxplots[:,0],density=True,bins=100,alpha=0.4,color=(0,1,0))
plt.hist(data_DFE_boxplots[:,3],density=True,bins=100,alpha=0.4,color=(0,0.5,0))
plt.hist(data_DFE_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0,0.3,0))
plt.grid(False)
plt.xlim(min(data_DFE_boxplots[:,0]),max(data_DFE_boxplots[:,0]))
## END CHUNK





#THIS CHUNK IS FOR BOXPLOTS FOR BENEFICIAL FRACTION
boxprops_DFE = dict(linestyle='-', linewidth=1, color=[0,0.8,0])
meanprops_DFE =dict(linestyle='--', linewidth=1, color=[0,0.7,0])
whiskerprops_DFE =dict(linestyle='-', linewidth=1, color=[0,0.8,0])
medianprops_DFE = dict(linestyle='-', linewidth=0, color=[0,0.8,0])

data_DFE_boxplots = np.loadtxt(file_for_reading+"\DFEdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
number_of_ben_fe = np.zeros(len(data_DFE_boxplots[0]),dtype=int)
for j in range(len(number_of_ben_fe)):
    for i in range(len(data_DFE_boxplots)):
        if data_DFE_boxplots[i,j] >= 0:
            number_of_ben_fe[j] += 1

DFE_samples_per_walk = N*(B-1)
total_walks = int(len(data_DFE_boxplots)/DFE_samples_per_walk)
data_DFE_ben_frac = np.zeros((total_walks,len(data_DFE_boxplots[0])))

for mut in range(len(data_DFE_boxplots[0])):
    walk = 0
    while walk < total_walks:
        for i in range(DFE_samples_per_walk):
            s = data_DFE_boxplots[walk*DFE_samples_per_walk + i,mut]
            if s >= 0:
                data_DFE_ben_frac[walk,mut] += 1 
        walk+=1
data_DFE_ben_frac = data_DFE_ben_frac/DFE_samples_per_walk

plt.figure(dpi=150)
plt.xlabel("Substitutions")    
# plt.fill_between(mut_points, DFE_anal_quartiles[:,1],DFE_anal_quartiles[:,3], alpha=0.4,color=[0,1,0])
# plt.fill_between(mut_points, DFE_anal_quartiles[:,0],DFE_anal_quartiles[:,4], alpha=0.2,color=[0,1,0])
# plt.plot(mut_points,DFE_stats[0]+DFE_stats[1],'--',color=[0,0.7,0])
# plt.legend(['DFE'],loc='upper right')

for i in range(len(number_of_ben_fe)):
    data_DFE_ben_boxplot = beneficial_DFE_for_data(data_DFE_boxplots[:,i])
    DFE_box=plt.boxplot(data_DFE_ben_boxplot,positions=[mut_boxplots[i]],whis=(2.5, 97.5),widths=m/20,medianprops=medianprops_DFE,showmeans=(True),meanline=(True),showfliers=False,boxprops=boxprops_DFE,meanprops=meanprops_DFE,whiskerprops=whiskerprops_DFE)
    plt.setp(DFE_box["caps"], color=[0,0.8,0])
    plt.ylabel('Fitness effect (s)')
    plt.xticks(mut_boxplots)
    if i == 0:
        data_DFE_ben_boxplots_initial = data_DFE_ben_boxplot
    if i == 3:
        data_DFE_ben_boxplots_intermediate = data_DFE_ben_boxplot
    if i == 10:
        data_DFE_ben_boxplots_final = data_DFE_ben_boxplot

plt.yscale('log')
plt.yticks([10**(-5),10**(-4),10**(-3),10**(-2)])

quartiles_percentages = np.array([0.025,0.25,0.5,0.75,0.975])
steps_for_quartiles = 10**5
s_points_ben, DFE_ana_ben = computeAllDFE_ben(L_ana, M_ana, w_points, N, chebishev_interpolation)
DFE_cumulative_ben = np.zeros(DFE_ana_ben.shape)
for mut in range(m+1):
    splines_DFE = si.CubicSpline(s_points_ben[mut], DFE_ana_ben[mut])
    for j in range(number_points):
        DFE_cumulative_ben[mut,j], error = sp.quad(splines_DFE, s_points_ben[mut,0], s_points_ben[mut,j], args=(),epsabs=10**(-4), epsrel=10**(-4), limit=10**6)

DFE_anal_quartiles_ben = np.zeros((m+1,len(quartiles_percentages)))
for mut in range(m+1):
    
    for q in range(len(quartiles_percentages)):
        for s in np.linspace(s_points_ben[mut,0],s_points_ben[mut,-1],steps_for_quartiles):
            DFE_cum_s = chebishev_interpolation(s,s_points_ben[mut],DFE_cumulative_ben[mut])
            if DFE_cum_s >= quartiles_percentages[q]:
                DFE_anal_quartiles_ben[mut,q] = s
                break
plt.fill_between(mut_points, DFE_anal_quartiles_ben[:,1],DFE_anal_quartiles_ben[:,3], alpha=0.4,color=[0,1,0])
plt.fill_between(mut_points, DFE_anal_quartiles_ben[:,0],DFE_anal_quartiles_ben[:,4], alpha=0.2,color=[0,1,0])

DFE_mean_ben = np.zeros(m+1)
for mut in range(m+1):
    splines_DFE = si.CubicSpline(s_points_ben[mut], DFE_ana_ben[mut])
    DFE_mean_ben[mut], error = sp.quad(DFE_mean_integrand,0, s_points_ben[mut,-1], args=(splines_DFE),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    
plt.plot(mut_points,DFE_mean_ben,'--',color=[0,0.7,0])
## END OF CHUNK






## THIS CHUNK IS FOR THE BENEFICIAL FRACTION OF THE DFE NORMALIZED
plt.figure(dpi=150)
plt.plot([],[],color=(0.8,0.8,0.8))
plt.plot([],[],color=(0.5,0.5,0.5))
plt.plot([],[],color=(0.1,0.1,0.1))
plt.ylabel("Probability density")
plt.xlabel('Fitness effect, $s$')
plt.legend(('i = '+str(mut_boxplots[0]),'i = '+str(mut_boxplots[3]),'i = '+str(mut_boxplots[-1])))

# Get the detailed DFE initially
s_points_ben_detailed = np.linspace(s_points_ben[0,0],s_points_ben[0,-1],number_points_for_detailed)
splines_DFE = si.CubicSpline(s_points_ben[0], DFE_ana_ben[0])
DFE_ana_ben_detailed = splines_DFE(s_points_ben_detailed)
plt.plot(s_points_ben_detailed,DFE_ana_ben_detailed,color=(0,0.9,0))
# Get the detailed DFE at an intermediate step
s_points_ben_detailed = np.linspace(s_points_ben[60,0],s_points_ben[60,-1],number_points_for_detailed)
splines_DFE = si.CubicSpline(s_points_ben[60], DFE_ana_ben[60])
DFE_ana_ben_detailed = splines_DFE(s_points_ben_detailed)
plt.plot(s_points_ben_detailed,DFE_ana_ben_detailed,color=(0,0.5,0))
# Get the detailed DFE at the end
s_points_ben_detailed = np.linspace(s_points_ben[-1,0],s_points_ben[-1,-1],number_points_for_detailed)
splines_DFE = si.CubicSpline(s_points_ben[-1], DFE_ana_ben[-1])
DFE_ana_ben_detailed = splines_DFE(s_points_ben_detailed)
plt.plot(s_points_ben_detailed,DFE_ana_ben_detailed,color=(0,0.3,0))

plt.hist(data_DFE_ben_boxplots_initial,density=True,bins=100,alpha=0.4,color=(0,1,0))
plt.hist(data_DFE_ben_boxplots_intermediate,density=True,bins=100,alpha=0.4,color=(0,0.5,0))
plt.hist(data_DFE_ben_boxplots_final,density=True,bins=100,alpha=0.4,color=(0,0.3,0))
plt.xlim([0,0.02])
plt.xticks([0,0.005,0.01,0.015,0.02])
## END CHUNK



## THIS PART PLOT THE DISTRIBUTIONS/DFE 8 PANEL FIGURE
if B == 20:

    L_ana_final_detailed = np.zeros(len(w_points_detailed))
    M_ana_final_detailed = np.zeros(len(w_points_detailed))
    for i in range(len(w_points_detailed)):
        L_ana_final_detailed[i] = chebishev_interpolation(w_points_detailed[i],w_points,L_ana[-1])
        M_ana_final_detailed[i] = chebishev_interpolation(w_points_detailed[i],w_points,M_ana[-1])
    #beneficial fraction DFE
    s_points_ben_20, DFE_ana_ben_20 = computeAllDFE_ben(L_ana, M_ana, w_points, N, chebishev_interpolation)
    s_points_ben_detailed_20 = np.linspace(s_points_ben_20[-1,0],s_points_ben_20[-1,-1],number_points_for_detailed)
    splines_DFE = si.CubicSpline(s_points_ben_20[-1], DFE_ana_ben_20[-1])
    DFE_ana_ben_detailed_20 = splines_DFE(s_points_ben_detailed_20)
    
    B=2
    m=60
    data_L_boxplots_2 = np.loadtxt(file_for_reading+"\Ldistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    data_M_boxplots_2 = np.loadtxt(file_for_reading+"\Mdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    data_DFE_boxplots_2 = np.loadtxt(file_for_reading+"\DFEdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    L_ana_2,M_ana_2,LF_ana_2,MF_ana_2 = distributions_over_time(N,B,pop_size,m,w_points,chebishev_interpolation)
    L_ana_final_detailed_2 = np.zeros(len(w_points_detailed))
    M_ana_final_detailed_2 = np.zeros(len(w_points_detailed))
    for i in range(len(w_points_detailed)):
        L_ana_final_detailed_2[i] = chebishev_interpolation(w_points_detailed[i],w_points,L_ana_2[-1])
        M_ana_final_detailed_2[i] = chebishev_interpolation(w_points_detailed[i],w_points,M_ana_2[-1])
    #beneficial fraction DFE
    s_points_ben_2, DFE_ana_ben_2 = computeAllDFE_ben(L_ana_2, M_ana_2, w_points, N, chebishev_interpolation)
    s_points_ben_detailed_2 = np.linspace(s_points_ben_2[-1,0],s_points_ben_2[-1,-1],number_points_for_detailed)
    splines_DFE = si.CubicSpline(s_points_ben_2[-1], DFE_ana_ben_2[-1])
    DFE_ana_ben_detailed_2 = splines_DFE(s_points_ben_detailed_2)
    # beneficial fraction DFE for data
    data_DFE_ben_2 = beneficial_DFE_for_data(data_DFE_boxplots_2[:,-1])
    
    B='inf'
    m=500
    data_L_boxplots_inf = np.loadtxt(file_for_reading+"\Ldistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    data_M_boxplots_inf = np.loadtxt(file_for_reading+"\Mdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    data_DFE_boxplots_inf = np.loadtxt(file_for_reading+"\DFEdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    L_ana_inf,M_ana_inf,LF_ana_inf,MF_ana_inf = distributions_over_time(N,10**6,pop_size,m,w_points,chebishev_interpolation)
    L_ana_final_detailed_inf = np.zeros(len(w_points_detailed))
    M_ana_final_detailed_inf = np.zeros(len(w_points_detailed))
    for i in range(len(w_points_detailed)):
        L_ana_final_detailed_inf[i] = chebishev_interpolation(w_points_detailed[i],w_points,L_ana_inf[-1])
        M_ana_final_detailed_inf[i] = chebishev_interpolation(w_points_detailed[i],w_points,M_ana_inf[-1])
    #beneficial fraction DFE
    s_points_ben_inf, DFE_ana_ben_inf = computeAllDFE_ben(L_ana_inf, M_ana_inf, w_points, N, chebishev_interpolation)
    s_points_ben_inf, DFE_ana_ben_inf = computeAllDFE_ben(L_ana_inf, M_ana_inf, w_points, N, chebishev_interpolation)
    s_points_ben_detailed_inf = np.linspace(s_points_ben_inf[-1,0],s_points_ben_inf[-1,-1],number_points_for_detailed)
    splines_DFE = si.CubicSpline(s_points_ben_inf[-1], DFE_ana_ben_inf[-1])
    DFE_ana_ben_detailed_inf = splines_DFE(s_points_ben_detailed_inf)
    # beneficial fraction DFE for data
    data_DFE_ben_inf = beneficial_DFE_for_data(data_DFE_boxplots_inf[:,-1])
    
    B=4
    m=200
    data_L_boxplots_4 = np.loadtxt(file_for_reading+"\Ldistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    data_M_boxplots_4 = np.loadtxt(file_for_reading+"\Mdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    data_DFE_boxplots_4 = np.loadtxt(file_for_reading+"\DFEdistribution_N"+str(N)+"_P"+str(N*10)+"_B"+str(B)+"_boxplot.csv")
    L_ana_4,M_ana_4,LF_ana_4,MF_ana_4 = distributions_over_time(N,B,pop_size,m,w_points,chebishev_interpolation)
    Fitness_4,error = sp.quad(distributions_mean_integrand, 0, 1, args=(L_ana_4[-1], w_points, chebishev_interpolation),epsabs=10**(-8), epsrel=10**(-8), limit=10**6)
    L_ana_final_detailed_4 = np.zeros(len(w_points_detailed))
    M_ana_final_detailed_4 = np.zeros(len(w_points_detailed))
    for i in range(len(w_points_detailed)):
        L_ana_final_detailed_4[i] = chebishev_interpolation(w_points_detailed[i],w_points,L_ana_4[-1])
        M_ana_final_detailed_4[i] = chebishev_interpolation(w_points_detailed[i],w_points,M_ana_4[-1])
    #beneficial fraction DFE
    s_points_ben_4, DFE_ana_ben_4 = computeAllDFE_ben(L_ana_4, M_ana_4, w_points, N, chebishev_interpolation)
    s_points_ben_detailed_4 = np.linspace(s_points_ben_4[-1,0],s_points_ben_4[-1,-1],number_points_for_detailed)
    splines_DFE = si.CubicSpline(s_points_ben_4[-1], DFE_ana_ben_4[-1])
    DFE_ana_ben_detailed_4 = splines_DFE(s_points_ben_detailed_4)
    # beneficial fraction DFE for data
    data_DFE_ben_4 = beneficial_DFE_for_data(data_DFE_boxplots_4[:,-1])
    
    #start plotting
    fig, axs = plt.subplots(2,4,sharex=False,dpi=150, layout='constrained',figsize=(12,5))
    axs[1,3].plot([],[],color=(0,0.6,0))
    axs[1,3].plot([],[],color=(0,0,1))
    axs[1,3].plot([],[],color=(1,0,0))
    axs[1,3].legend(["DFE","$L_i$","$M_i$"])
    
    axs[1,0].hist(data_L_boxplots_2[:,-1],density=True,bins=100,alpha=0.4,color=(0,0,0.8))
    axs[1,0].plot(w_points_detailed,L_ana_final_detailed_2,color=(0,0,0.8))
    axs[1,0].plot(w_points_detailed,M_ana_final_detailed_2,color=(0.8,0,0))
    axs[1,0].hist(data_M_boxplots_2[:,-1],density=True,bins=100,alpha=0.4,color=(0.8,0,0))
    
    axs[1,2].hist(data_L_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0,0,0.8))
    axs[1,2].plot(w_points_detailed,L_ana_final_detailed,color=(0,0,0.8))
    axs[1,2].plot(w_points_detailed,M_ana_final_detailed,color=(0.8,0,0))
    axs[1,2].hist(data_M_boxplots[:,-1],density=True,bins=100,alpha=0.4,color=(0.8,0,0))
    
    axs[1,1].hist(data_L_boxplots_4[:,-1],density=True,bins=100,alpha=0.4,color=(0,0,0.8))
    axs[1,1].plot(w_points_detailed,L_ana_final_detailed_4,color=(0,0,0.8))
    axs[1,1].plot(w_points_detailed,M_ana_final_detailed_4,color=(0.8,0,0))
    axs[1,1].hist(data_M_boxplots_4[:,-1],density=True,bins=100,alpha=0.4,color=(0.8,0,0))
    
    axs[1,3].hist(data_L_boxplots_inf[:,-1],density=True,bins=100,alpha=0.4,color=(0,0,0.8))
    axs[1,3].plot(w_points_detailed,L_ana_final_detailed_inf,color=(0,0,0.8))
    axs[1,3].plot(w_points_detailed,M_ana_final_detailed_inf,color=(0.8,0,0))
    axs[1,3].hist(data_M_boxplots_inf[:,-1],density=True,bins=100,alpha=0.4,color=(0.8,0,0))
    
    axs[0,0].plot(s_points_ben_detailed_2,DFE_ana_ben_detailed_2,color=(0,0.6,0))
    axs[0,0].hist(data_DFE_ben_2,density=True,bins=100,alpha=0.4,color=(0,0.6,0))
    axs[0,0].set_xlim([min(data_DFE_ben_2),0.004])
    
    
    axs[0,2].plot(s_points_ben_detailed_20,DFE_ana_ben_detailed_20,color=(0,0.6,0))
    axs[0,2].hist(data_DFE_ben_boxplots_final,density=True,bins=100,alpha=0.4,color=(0,0.6,0))
    axs[0,2].set_xlim([min(data_DFE_ben_boxplots_final),0.004])
    
    axs[0,1].plot(s_points_ben_detailed_4,DFE_ana_ben_detailed_4,color=(0,0.6,0))
    axs[0,1].hist(data_DFE_ben_4,density=True,bins=100,alpha=0.4,color=(0,0.6,0))
    axs[0,1].set_xlim([min(data_DFE_ben_4),0.004])
    
    axs[0,3].plot(s_points_ben_detailed_inf,DFE_ana_ben_detailed_inf,color=(0,0.6,0))
    axs[0,3].hist(data_DFE_ben_inf,density=True,bins=100,alpha=0.4,color=(0,0.6,0))
    axs[0,3].set_xlim([min(data_DFE_ben_inf),0.004])
    
    
    axs[1,1].set_xlabel("Fitness contribution")
    axs[1,1].xaxis.set_label_coords(1, -0.12)
    axs[1,0].set_ylabel("Probability density")
    axs[0,0].set_title("$B=2$")
    axs[0,1].set_title("$B=4$")
    axs[0,2].set_title("$B=20$")
    axs[0,3].set_title("$B=$Infinite")
    
    axs[0,1].set_xlabel("Fitness effect, $s$")
    axs[0,1].xaxis.set_label_coords(1, -0.13)
    axs[0,0].set_ylabel("Probability density")
    
    ## END
    
















