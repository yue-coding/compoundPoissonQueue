from timeit import default_timer as timer

# Uses the first fluid approximation method to calculate the number of customers in the system.
# Note: The return value is a tuple:
#       the first element is a list of x values,
#       the second element is the number of customers in the system at corresponding time x (using same index)
def fluid(initial_queue_length,ArrivalRateSeries, BulksizeSeries, ServiceRate):
    start = timer()
    BatchArrivalRateSeries=[]
   
    for i in range(0,len(BulksizeSeries)):
        BatchArrivalRateSeries.append(ArrivalRateSeries[i]*BulksizeSeries[i])

    differenceSeries=list(map(lambda x: x-ServiceRate, BatchArrivalRateSeries))
    
    QueueLengthSeries =[initial_queue_length]
    x_axis=[0]
    for i in range(1,len(ArrivalRateSeries)+1):
         #if the queue length hit zero in between intervals, then it is necesarry to calculate
         #the exact x when y=0
        if differenceSeries[i-1] <0 and QueueLengthSeries[i-1]>0:
            x_value=QueueLengthSeries[i-1]/abs(differenceSeries[i-1])
            if x_value<1:
                QueueLengthSeries.append(0)
                x_axis.append(i-1+x_value)
        QueueLengthSeries.append(max(differenceSeries[i-1] + QueueLengthSeries[i-1],0))
        x_axis.append(i)
    end = timer()
    print(f'runtime 1st fluid approximation: {end - start}')
    
    return (x_axis,QueueLengthSeries)

# Uses the second fluid approximation method to calculate the number of customers in the system.
def second_fluid(initial_queue_length,ArrivalRateSeries, BulksizeSeries, ServiceRate, EulerStepsPerInterval):
    #here we use Euler's method to approximate a differential equation from Jimenez and Koole
    start = timer()
    BatchArrivalRateSeries=[]
    for i in range(0,len(BulksizeSeries)):
        BatchArrivalRateSeries.append(ArrivalRateSeries[i]*BulksizeSeries[i])
    
    differenceSeries=[]
    QueueLengthSeries =[initial_queue_length]
    for i in range(0,len(ArrivalRateSeries)*EulerStepsPerInterval-1):
        differenceSeries.append(BatchArrivalRateSeries[i//EulerStepsPerInterval]-min(QueueLengthSeries[i],1)*ServiceRate)
        QueueLengthSeries.append(QueueLengthSeries[i]+differenceSeries[i]/EulerStepsPerInterval)
    end = timer()
    print(f'runtime 2nd fluid approximation: {end - start}')
    return QueueLengthSeries
