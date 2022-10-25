import numpy as np
import math
from scipy.stats import poisson
from scipy.stats import randint
from timeit import default_timer as timer

# Calculates the expected value of the number of customers in the system from a state probability vector,
# where the index is the number of customers in the system and the values are the probabilities.
def expectedValue(arr):
    weisum=0            
    for i in range(0,len(arr)):
        weisum+=i*arr[i]
    return weisum
    
# Variable to cache partitions results, to prevent calculating the same partitions multiple times.
partitions_cache = {}

# Calculates the partitions for the given parameters.
def partitions(n, minEl, maxEl, maxDepth):
    if maxDepth == 0:
        return []
    maxEl = min(n, maxEl)
    key = f'{n},{minEl},{maxEl},{maxDepth}'
    if key in partitions_cache:
        return partitions_cache[key]
    result = []
    for k in range(minEl, maxEl + 1):
        if k == n:
            result.append([k])
        else:
            result.extend([
                [k] + partitionSet
                for partitionSet
                in partitions(n-k, max(k, minEl), maxEl, maxDepth - 1)
            ])
    partitions_cache[key] = result
    return result

# Uses the DTA to calculate the number of customers in the system with deterministic batch sizes.
def DTA_deterministic_bulk_size(initial_queue_length, ArrivalRateSeries, BulksizeSeries, ServiceRate, AmountOfIntervalPerMinute, K):
    start = timer()

    #first make the arrival rate values smaller according to how many intervals there are
    SlicedArrivalRateSeries=list(map(lambda x: x/AmountOfIntervalPerMinute, ArrivalRateSeries))
    SlicedServiceRate=ServiceRate/AmountOfIntervalPerMinute
    
    #Probability Pai of K new arrivals to the system Poisson-distributed
    #Pai has two indices: time and k, is therefore defined as a 2 dimentional array
    #the range of t is the size of A_t, the range of k is K
    rows, cols = (len(ArrivalRateSeries), K) 
    pai_t_k = [[0 for i in range(0,cols)] for j in range(0,rows)]
    
    for t in range(0,len(ArrivalRateSeries)):
        for k in range(0,K-1):
            if k*BulksizeSeries[t]<K-1:
                pai_t_k[t][k*BulksizeSeries[t]]=poisson.pmf(k,SlicedArrivalRateSeries[t])
        pai_t_k[t][K-1]=max(1 - sum(pai_t_k[t]),0)   
    
    #Probability D of k customer leave the system
    #probability S of serving k people, S is not time-dependent, so only one dimentional
    S_k=[]
    for k in range(0,K):
        #S_k.append(SlicedServiceRate**k/math.factorial(k)*math.exp(-SlicedServiceRate))
        S_k.append(poisson.pmf(k,SlicedServiceRate))
    
    #But the number of departture is not necissarily the number of people served drawing from the poisson distribution
    #for example we may be able to serve 3 people, but there is only 1 in the queue.
    #therefore a seperate variable is defined d_q_k, q stands for the number of people in the system
    D_q_k = [[0 for i in range(0,K)] for j in range(0,K)]
    for q in range(0,K):
        for k in range(0,K):
            if q > k:
                D_q_k[q][k]=S_k[k]
            elif q == k:
                D_q_k[q][k]=1-sum(S_k[:k])
            else:
                D_q_k[q][k]=0
                
    #initialize the P_t as an empty queue 
    P_t=np.zeros(K)
    P_t[initial_queue_length]=1

    #the performance measure DTA_WIP_ave 
    DTA_WIP_ave=[0 for i in range(0,len(SlicedArrivalRateSeries)*AmountOfIntervalPerMinute)]
    

    #build the transition matrix. The matrix is 2 dimentional, queue length before the transition, queue length 
    #after the transition. P_i_j
    #also need to define a P_t probability vector

    for t in range(0,len(ArrivalRateSeries)):
        P_i_j= [[0 for i in range(0,K)] for j in range(0,K)]
        for i in range(0,K):
            for j in range(0,K):
                for counter in range(0,i+1):
                    if counter+j-i>=0:
                        P_i_j[i][j]+=pai_t_k[t][counter+j-i]*D_q_k[i][counter]
                    else:
                        P_i_j[i][j]+=0
        P_i_j_numpy=np.asarray(P_i_j)
        #P_t is the array multiplication of the transition matrix and the last P_t
        #this step need to be repeat n times. n is the time we splice a time unit
        for slicedTimeInverval in range(0,AmountOfIntervalPerMinute):
            P_t=np.matmul(P_t,P_i_j_numpy)
            
            #the weighted average probability of the queue length is needed for drawing the graph
            DTA_WIP_ave[t*AmountOfIntervalPerMinute+slicedTimeInverval]=expectedValue(P_t)
    end = timer()
    DTA_WIP_ave.insert(initial_queue_length,0)
    print(f'the run time is {end - start}')
   
    return DTA_WIP_ave

# Uses the DTA to calculate the number of customers in the system with uniformly distributed batch sizes.
def DTA_Uni_bulk_size(initial_queue_length, ArrivalRateSeries, BulksizeSeries, RangeOfBulksize, ServiceRate, AmountOfIntervalPerMinute, K):
    start = timer()
    #first make the arrival rate values smaller according to how many intervals there are
    SlicedArrivalRateSeries=list(map(lambda x: x/AmountOfIntervalPerMinute, ArrivalRateSeries))
    SlicedServiceRate=ServiceRate/AmountOfIntervalPerMinute
    #Probability Pai of K new arrivals to the system Poisson-distributed
    #Pai has two indices: time and k, is therefore defined as a 2 dimentional array
    #the range of t is the size of A_t, the range of k is K
    rows, cols = (len(ArrivalRateSeries), K) 
    pai_t_k =[[0 for i in range(0,cols)] for j in range(0,rows)]
    
    # Cache results for rows and poisson probabilities by set of parameters, so we don't calculate the same thing multiple times.
    poisson_cache = {}
    row_cache = {}
    
    for t in range(0,len(ArrivalRateSeries)):
        key = f'{ArrivalRateSeries[t]},{BulksizeSeries[t]},{RangeOfBulksize[t]}'
        if key in row_cache:
            pai_t_k[t] = row_cache[key]
            continue
        BatchSizeProbabilities=[
            randint.pmf(k_batch,BulksizeSeries[t]-RangeOfBulksize[t],BulksizeSeries[t]+RangeOfBulksize[t]+1)
            for k_batch
            in range(0,BulksizeSeries[t]+RangeOfBulksize[t]+1)
        ]
        
        # calculate probabilities for number of groups arriving in one small time-step
        minBulkSize = BulksizeSeries[t] - RangeOfBulksize[t]
        maxBulkSize = BulksizeSeries[t] + RangeOfBulksize[t]
        groupArrivalProbs = []
        groupArrivalProbsSum = 0
        while groupArrivalProbsSum < 0.999999999 and len(groupArrivalProbs) * minBulkSize < K -1:
            poissonValue = 0
            poisson_key = f'{len(groupArrivalProbs)},{SlicedArrivalRateSeries[t]}'
            if poisson_key in poisson_cache:
                poissonValue = poisson_cache[poisson_key]
            else:
                poissonValue = poisson.pmf(len(groupArrivalProbs),SlicedArrivalRateSeries[t])
                poisson_cache[poisson_key] = poissonValue
            groupArrivalProbs.append(poissonValue)
            groupArrivalProbsSum += poissonValue
        groupArrivalProbs.append(1 - groupArrivalProbsSum)
        
        # generate pai_t_k
        pai_t_k[t][0] = groupArrivalProbs[0]
        for k in range(1,K-1):
            for partition in partitions(k,minBulkSize,maxBulkSize,len(groupArrivalProbs)-1):
                pai_t_k[t][k] += (
                    groupArrivalProbs[len(partition)] *
                    np.prod([BatchSizeProbabilities[i] for i in partition]) *
                    math.factorial(len(partition))
                )
          
        pai_t_k[t][K-1]=max(1 - sum(pai_t_k[t]),0)
        row_cache[key] = pai_t_k[t]
    
    #Probability D of k customer leave the system
    #probability S of serving k people, S is not time-dependent, so only one dimentional
    S_k=[]
    for k in range(0,K):
    #S_k.append(SlicedServiceRate**k/math.factorial(k)*math.exp(-SlicedServiceRate))
        S_k.append(poisson.pmf(k,SlicedServiceRate))
    
    #But the number of departure is not necissarily the number of people served drawing from the poisson distribution
    #for example we may be able to serve 3 people, but there is only 1 in the queue.
    #therefore a seperate variable is defined d_q_k, q stands for the number of people in the system
    D_q_k = [[0 for i in range(0,K)] for j in range(0,K)]
    for q in range(0,K):
        for k in range(0,K):
            if q > k:
                D_q_k[q][k]=S_k[k]
            elif q == k:
                D_q_k[q][k]=1-sum(S_k[:k])
            else:
                D_q_k[q][k]=0
                
    #initialize the P_t as an empty queue 
    P_t=np.zeros(K)
    P_t[initial_queue_length]=1

    #the performance measure DTA_WIP_ave 
    DTA_WIP_ave=[0 for i in range(0,len(SlicedArrivalRateSeries)*AmountOfIntervalPerMinute)]

    #build the transition matrix. The matrix is 2 dimentional, queue length before the transition, queue length 
    #after the transition. P_i_j
    #also need to define a P_t probability vector
   
    for t in range(0,len(ArrivalRateSeries)):
        P_i_j= [[0 for i in range(0,K)] for j in range(0,K)]
        for i in range(0,K):
            for j in range(0,K):
                for counter in range(0,i+1):
                    if counter+j-i>=0:
                        P_i_j[i][j]+=pai_t_k[t][counter+j-i]*D_q_k[i][counter]
                    else:
                        P_i_j[i][j]+=0
        P_i_j_numpy=np.asarray(P_i_j)    
           
        #P_t is the array multiplication of the transition matrix and the last P_t
        #this step need t rval in range(0,AmountOfIntervalPerMinute):
        for slicedTimeInverval in range(0,AmountOfIntervalPerMinute):
            P_t=np.matmul(P_t,P_i_j_numpy)
            #the weighted average probability of the queue length is needed for drawing the graph
            DTA_WIP_ave[t*AmountOfIntervalPerMinute+slicedTimeInverval]=expectedValue(P_t)
    DTA_WIP_ave.insert(initial_queue_length,0)
    end = timer()
    print(f'the run time of DTA with stochastic batch size is {end-start}')
    return DTA_WIP_ave

# Calculates the probability of having more than K customers in the system after a small time interval.
# This is needed for determining the minimal K for a given tolerance level.
def DTA_deterministic_bulk_size_Error_Calculation(initial_queue_length, ArrivalRateSeries, BulksizeSeries, ServiceRate, AmountOfIntervalPerMinute, K):
    start = timer()

    #first make the arrival rate values smaller according to how many intervals there are
    SlicedArrivalRateSeries=list(map(lambda x: x/AmountOfIntervalPerMinute, ArrivalRateSeries))
    SlicedServiceRate=ServiceRate/AmountOfIntervalPerMinute

    rows, cols = (len(ArrivalRateSeries), K) 
    pai_t_k = [[0 for i in range(0,cols)] for j in range(0,rows)]
    
    for t in range(0,len(ArrivalRateSeries)):
        for k in range(0,K-1):
            if k*BulksizeSeries[t]<K-1:
                pai_t_k[t][k*BulksizeSeries[t]]=poisson.pmf(k,SlicedArrivalRateSeries[t])
        pai_t_k[t][K-1]=max(1 - sum(pai_t_k[t]),0)   
    
    #Probability D of k customer leave the system
    #probability S of serving k people, S is not time-dependent, so only one dimentional
    S_k=[]
    for k in range(0,K):
        S_k.append(poisson.pmf(k,SlicedServiceRate))
    
    #But the number of departture is not necissarily the number of people served drawing from the poisson distribution
    #for example we may be able to serve 3 people, but there is only 1 in the queue.
    #therefore a seperate variable is defined d_q_k, q stands for the number of people in the system
    D_q_k = [[0 for i in range(0,K)] for j in range(0,K)]
    for q in range(0,K):
        for k in range(0,K):
            if q > k:
                D_q_k[q][k]=S_k[k]
            elif q == k:
                D_q_k[q][k]=1-sum(S_k[:k])
            else:
                D_q_k[q][k]=0
                
    #initialize the P_t as an empty queue 
    P_t=np.zeros(K)
    P_t[initial_queue_length]=1
    #build the transition matrix. The matrix is 2 dimentional, queue length before the transition, queue length 
    #after the transition. P_i_j
    #also need to define a P_t probability vector
    ErrorProbability=0
    for t in range(0,len(ArrivalRateSeries)):
        P_i_j= [[0 for i in range(0,K)] for j in range(0,K)]
        for i in range(0,K):
            for j in range(0,K):
                for counter in range(0,i+1):
                    if counter+j-i>=0:
                        P_i_j[i][j]+=pai_t_k[t][counter+j-i]*D_q_k[i][counter]
                    else:
                        P_i_j[i][j]+=0
        P_i_j_numpy=np.asarray(P_i_j)        
        #P_t is the array multiplication of the transition matrix and the last P_t
        #this step need to be repeat n times. n is the time we splice a time unit
        for slicedTimeInverval in range(0,AmountOfIntervalPerMinute):
            P_t=np.matmul(P_t,P_i_j_numpy)    
        ErrorProbability=max(ErrorProbability,P_t[K-1])
    end = timer()
    return ErrorProbability

# Determines the minimal K for a given tolerance level.
# NOTE: The reutnr value is a tuple. The first element is the determined K, the second element is the actual error with that K.
def findK_DTA_deterministic_bulk_size(Tolerance, initial_queue_length, arrivalRateSeries, BulksizeSeries, serviceRate, AmountOfIntervalPerMinute):
    k=initial_queue_length+1
    error = 1
    while(error > Tolerance):
        error = DTA_deterministic_bulk_size_Error_Calculation(initial_queue_length,arrivalRateSeries, BulksizeSeries, serviceRate,AmountOfIntervalPerMinute,k)
        k+=1 
    return (k, error)
