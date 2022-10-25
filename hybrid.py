import numpy as np
import math
from scipy.stats import poisson
from timeit import default_timer as timer

# Uses the fluid approximation with idle rate correction to calculate the number of customers in the system.
def fluid_with_idleRate(initial_queue_length, ArrivalRateSeries, BulksizeSeries, ServiceRate, idleRate):
    start = timer()
    BatchArrivalRateSeries=[]
    numberOfIntervalsPerTimeUnit=len(idleRate)/len(ArrivalRateSeries)
    
    for i in range(0,len(BulksizeSeries)):
        BatchArrivalRateSeries.append(ArrivalRateSeries[i]*BulksizeSeries[i])

    differenceSeries=list(map(lambda x: (x-ServiceRate)/numberOfIntervalsPerTimeUnit, BatchArrivalRateSeries))
    differenceSeries=np.repeat(differenceSeries,numberOfIntervalsPerTimeUnit)
    
    QueueLengthSeries =[initial_queue_length]
    for i in range(0, len(differenceSeries)-1):
         #if the queue length hit zero in between intervals, then it is necesarry to calculate
         #the exact x when y=0
        prevQueueLength = QueueLengthSeries[-1]
        stepDiff = differenceSeries[i] + idleRate[i]*ServiceRate/numberOfIntervalsPerTimeUnit
        #print(f'idleRate({i}):{idleRate[i]}')
        #print(f'stepDiff({i}):{stepDiff}')
        QueueLengthSeries.append(max(prevQueueLength + stepDiff, 0))
        #print(f'queueLength({i}):{QueueLengthSeries[-1]}')
    end = timer()
    print(f'runtime hybrid fluid approximation: {end - start}')
    
    return QueueLengthSeries

# Uses the DTA to calculate the idle rate for systems with deterministic batch size.
def DTA_deterministic_bulk_size_idletime(initial_queue_length, ArrivalRateSeries, BulksizeSeries, ServiceRate, AmountOfIntervalPerMinute, K):
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
    DTA_PofIdle=[0 for i in range(0,len(SlicedArrivalRateSeries)*AmountOfIntervalPerMinute)]
    

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
            DTA_PofIdle[t*AmountOfIntervalPerMinute+slicedTimeInverval]=P_t[0]
    end = timer()
    print(end - start)
    return DTA_PofIdle

# Performs the hybrid approach by first using DTA to obtain an apporoximation of the idle rate
# and then using that to run the fluid approximation with idle rate correction.
def hybrid_DTA_and_fluid(initial_queue_length, ArrivalRateSeries, BulksizeSeries, ServiceRate, AmountOfIntervalPerMinute, K):
    idleRate = DTA_deterministic_bulk_size_idletime(initial_queue_length,ArrivalRateSeries, BulksizeSeries, ServiceRate, AmountOfIntervalPerMinute, K)
    return fluid_with_idleRate(initial_queue_length,ArrivalRateSeries, BulksizeSeries, ServiceRate, idleRate)
