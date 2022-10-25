import numpy as np
import math
import time
import multiprocessing
import concurrent.futures

# This function works for uniformly distributed batch size.
# It can also be used for deterministic batch size by setting rangeOfBatchSize to 0 for all time intervals.
# This function calculates the number of customers in the system.
def SimulationStochasticBatch(initialQueue,
                              arrivalRate,
                              BatchAverageSize,
                              rangeOfBatchSize,
                              serviceRate,
                              numberOfIterations,
                              NumberOfStepsPerUnit,
                              threadNum = -1):
    prefix = ""
    if threadNum != -1:
        prefix = f'Thread {threadNum}: '
        print(f'Thread {threadNum} has started.')
    startTime = time.perf_counter()
    lastElapsedTime = 0
    numberOfOutPut=len(arrivalRate)*NumberOfStepsPerUnit+1
    AveSimulationWIP=[0 for i in range(numberOfOutPut)]
    AveSimulationWIP[0]=initialQueue
    for i in range(0,numberOfIterations):
        arrivalTimes=[]
        serviceTimes=[]
        batchSize=[]
        j=0
        while j < len(arrivalRate):
            nextArrivalTime = j + np.random.exponential(1/arrivalRate[int(j)])
            while int(j) < int(nextArrivalTime) and j + 1 < len(arrivalRate):
                j = int(j) + 1
                nextArrivalTime = j + np.random.exponential(1/arrivalRate[j])
            arrivalTimes.append(nextArrivalTime)
            if nextArrivalTime >= len(arrivalRate):
                break
           
            batchLow=BatchAverageSize[int(nextArrivalTime)]-rangeOfBatchSize[int(nextArrivalTime)]
            batchHigh=BatchAverageSize[int(nextArrivalTime)]+rangeOfBatchSize[int(nextArrivalTime)]+1
            batchSize.append(np.random.randint(batchLow,batchHigh))
            j = nextArrivalTime
       
        queueLength = initialQueue
        t = 0
        nextServiceTime = math.inf;
        nextArrivalTime = arrivalTimes[0]
        arrivalsIndex = 0
        observationTimes = [(x + 1) / NumberOfStepsPerUnit for x in range(numberOfOutPut)]
        observationIndex = 0
        while t <= len(arrivalRate):
            if t == observationTimes[observationIndex]:
                observationIndex += 1
                AveSimulationWIP[observationIndex] = (
                    (AveSimulationWIP[observationIndex] * i + queueLength) / (i + 1)
                )
            if t == nextArrivalTime:
                queueLength += batchSize[arrivalsIndex]
                arrivalsIndex += 1
                nextArrivalTime = arrivalTimes[arrivalsIndex]
            if t == nextServiceTime:
                queueLength -= 1;
                nextServiceTime = math.inf;
            if queueLength > 0 and (nextServiceTime == math.inf):
                nextServiceTime = t + np.random.exponential(1/serviceRate)
            t = min(observationTimes[observationIndex], nextServiceTime, nextArrivalTime)
        elapsedTime = int(time.perf_counter() - startTime)
        if elapsedTime > lastElapsedTime or i == numberOfIterations - 1:
            numDone = i + 1
            print(
                f'\r{prefix}Progress (after {elapsedTime}s): {numDone}/{numberOfIterations} ({int(100 * (numDone) / numberOfIterations)}%)',
                                   end='')
            lastElapsedTime = elapsedTime
    print(f'\n{prefix}Running time: {time.perf_counter() - startTime}s')
    if threadNum != -1:
        print(f'Thread {threadNum} has finished.')
    return AveSimulationWIP

# This function works for uniformly distributed batch size.
# It can also be used for deterministic batch size by setting rangeOfBatchSize to 0 for all time intervals.
# This function calculates the idle rate.
def SimulationStochasticBatchIdleRate(initialQueue,
                              arrivalRate,
                              BatchAverageSize,
                              rangeOfBatchSize,
                              serviceRate,
                              numberOfIterations,
                              NumberOfStepsPerUnit,
                              threadNum = -1):
    prefix = ""
    if threadNum != -1:
        prefix = f'Thread {threadNum}: '
        print(f'Thread {threadNum} has started.')
    startTime = time.perf_counter()
    lastElapsedTime = 0
    numberOfOutPut=len(arrivalRate)*NumberOfStepsPerUnit+1
    AveSimulationIdle=[0 for i in range(numberOfOutPut)]
    AveSimulationIdle[0]=1 if initialQueue == 0 else 0
    for i in range(0,numberOfIterations):
        arrivalTimes=[]
        serviceTimes=[]
        batchSize=[]
        j=0
        while j < len(arrivalRate):
            nextArrivalTime = j + np.random.exponential(1/arrivalRate[int(j)])
            while int(j) < int(nextArrivalTime) and j + 1 < len(arrivalRate):
                j = int(j) + 1
                nextArrivalTime = j + np.random.exponential(1/arrivalRate[j])
            arrivalTimes.append(nextArrivalTime)
            if nextArrivalTime >= len(arrivalRate):
                break
           
            batchLow=BatchAverageSize[int(nextArrivalTime)]-rangeOfBatchSize[int(nextArrivalTime)]
            batchHigh=BatchAverageSize[int(nextArrivalTime)]+rangeOfBatchSize[int(nextArrivalTime)]+1
            batchSize.append(np.random.randint(batchLow,batchHigh))
            j = nextArrivalTime
       
        queueLength = initialQueue
        t = 0
        nextServiceTime = math.inf;
        nextArrivalTime = arrivalTimes[0]
        arrivalsIndex = 0
        observationTimes = [(x + 1) / NumberOfStepsPerUnit for x in range(numberOfOutPut)]
        observationIndex = 0
        while t <= len(arrivalRate):
            if t == observationTimes[observationIndex]:
                observationIndex += 1
                isIdle = 1 if queueLength == 0 else 0
                AveSimulationIdle[observationIndex] = (
                    (AveSimulationIdle[observationIndex] * i + isIdle) / (i + 1)
                )
            if t == nextArrivalTime:
                queueLength += batchSize[arrivalsIndex]
                arrivalsIndex += 1
                nextArrivalTime = arrivalTimes[arrivalsIndex]
            if t == nextServiceTime:
                queueLength -= 1;
                nextServiceTime = math.inf;
            if queueLength > 0 and (nextServiceTime == math.inf):
                nextServiceTime = t + np.random.exponential(1/serviceRate)
            t = min(observationTimes[observationIndex], nextServiceTime, nextArrivalTime)
        elapsedTime = int(time.perf_counter() - startTime)
        if elapsedTime > lastElapsedTime or i == numberOfIterations - 1:
            numDone = i + 1
            print(
                f'\r{prefix}Progress (after {elapsedTime}s): {numDone}/{numberOfIterations} ({int(100 * (numDone) / numberOfIterations)}%)',
                                   end='')
            lastElapsedTime = elapsedTime
    print(f'\n{prefix}Running time: {time.perf_counter() - startTime}s')
    if threadNum != -1:
        print(f'Thread {threadNum} has finished.')
    return AveSimulationIdle

# This function runs the simulation (SimulationStochasticBatch) across multiple processes to make use of multi-core cpus.
# This speeds up the simulation significantly on multi-core cpus without sacrificing accuracy.
# NOTE: To run multi-threaded in Jupiter notebook, the simulation function (SimulationStochasticBatch) has to be imported from a .py file.
def FastSimulationStochasticBatch(initialQueue,
                              arrivalRate,
                              BatchAverageSize,
                              rangeOfBatchSize,
                              serviceRate,
                              numberOfIterations,
                              NumberOfStepsPerUnit):
    startTime = time.perf_counter()
    numThreads = multiprocessing.cpu_count()
    iterationsPerThread = []
    divisor = numThreads
    outstandingIterations = numberOfIterations
    while divisor > 0:
        iterationsNext = outstandingIterations // divisor
        iterationsPerThread.append(iterationsNext)
        outstandingIterations -= iterationsNext
        divisor -= 1
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(SimulationStochasticBatch,
                            initialQueue,
                            arrivalRate,
                            BatchAverageSize,
                            rangeOfBatchSize,
                            serviceRate,
                            numIterations,
                            NumberOfStepsPerUnit,
                            i)
            for i, numIterations in enumerate(iterationsPerThread)]
        results = [f.result() for f in futures]
    
    numberOfOutPut=len(arrivalRate)*NumberOfStepsPerUnit+1
    AveSimulationWIP=[0 for i in range(numberOfOutPut)]
    collectedIterations = 0
    for i, result in enumerate(results):
        numIterations = iterationsPerThread[i]
        for j in range(numberOfOutPut):
            AveSimulationWIP[j] = (AveSimulationWIP[j] * collectedIterations +
                                   result[j] * numIterations) / (collectedIterations + numIterations)
        collectedIterations += numIterations
    print(f'Total running time: {time.perf_counter() - startTime}s')
    return AveSimulationWIP

# This function runs the simulation (SimulationStochasticBatchIdleRate) across multiple processes to make use of multi-core cpus.
# This speeds up the simulation significantly on multi-core cpus without sacrificing accuracy.
# NOTE: To run multi-threaded in Jupiter notebook, the simulation function (SimulationStochasticBatchIdleRate) has to be imported from a .py file.
def FastSimulationStochasticBatchIdleRate(initialQueue,
                              arrivalRate,
                              BatchAverageSize,
                              rangeOfBatchSize,
                              serviceRate,
                              numberOfIterations,
                              NumberOfStepsPerUnit):
    startTime = time.perf_counter()
    numThreads = multiprocessing.cpu_count()
    iterationsPerThread = []
    divisor = numThreads
    outstandingIterations = numberOfIterations
    while divisor > 0:
        iterationsNext = outstandingIterations // divisor
        iterationsPerThread.append(iterationsNext)
        outstandingIterations -= iterationsNext
        divisor -= 1
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(SimulationStochasticBatchIdleRate,
                            initialQueue,
                            arrivalRate,
                            BatchAverageSize,
                            rangeOfBatchSize,
                            serviceRate,
                            numIterations,
                            NumberOfStepsPerUnit,
                            i)
            for i, numIterations in enumerate(iterationsPerThread)]
        results = [f.result() for f in futures]
    
    numberOfOutPut=len(arrivalRate)*NumberOfStepsPerUnit+1
    AveSimulationIdle=[0 for i in range(numberOfOutPut)]
    collectedIterations = 0
    for i, result in enumerate(results):
        numIterations = iterationsPerThread[i]
        for j in range(numberOfOutPut):
            AveSimulationIdle[j] = (AveSimulationIdle[j] * collectedIterations +
                                   result[j] * numIterations) / (collectedIterations + numIterations)
        collectedIterations += numIterations
    print(f'Total running time: {time.perf_counter() - startTime}s')
    return AveSimulationIdle
