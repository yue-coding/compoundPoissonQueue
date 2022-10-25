import dta
import fluid
import hybrid
import simulation

if __name__ ==  '__main__':
    
    # system parameters
    initialWIP = 10
    arrivalRateSeries =     [1,1,2,2,3,4,4,5,1,1]
    bulksizeSeries =        [3,3,3,2,2,2,3,3,3,2]
    rangeOfBulkSizeSeries = [1,1,2,1,1,0,0,2,1,1]
    serviceRate = 8
    
    # approximation parameters
    simulation_numberOfIterations = 1000000
    simulation_numberOfSamplesPerUnit = 10
    dta_numberOfStepsPerUnit = 1000
    dta_toleranceForK = 0.00001
    secondFluid_eulerStepsPerInterval = 1000
    
    # using fluid
    print("Running first fluid.")
    firstFluidResults = fluid.fluid(initialWIP, arrivalRateSeries, bulksizeSeries, serviceRate)
    print("First fluid results (results are a tuple: first element is x_values, second element is number of customers in the system at corresponding time x):")
    print(firstFluidResults)
    
    print("Running second fluid.")
    secondFluidResults = fluid.second_fluid(initialWIP, arrivalRateSeries, bulksizeSeries, serviceRate, secondFluid_eulerStepsPerInterval)
    print("Second fluid results:")
    print(secondFluidResults)
    
    # using DTA
    print("Finding K for DTA.")
    dta_K = dta.findK_DTA_deterministic_bulk_size(dta_toleranceForK, initialWIP, arrivalRateSeries, bulksizeSeries, serviceRate, dta_numberOfStepsPerUnit)[0]
    print(f"K for DTA is: {dta_K}")
    
    print("Running dta for deterministic bulk size.")
    dtaDeterministicResults = dta.DTA_deterministic_bulk_size(initialWIP, arrivalRateSeries, bulksizeSeries, serviceRate, dta_numberOfStepsPerUnit, dta_K)
    print("DTA (deterministic bulk size) results:")
    print(dtaDeterministicResults)
    
    print("Running dta for stochastic bulk size.")
    dtaStochasticResults = dta.DTA_Uni_bulk_size(initialWIP, arrivalRateSeries, bulksizeSeries, rangeOfBulkSizeSeries, serviceRate, dta_numberOfStepsPerUnit, dta_K)
    print("DTA (stochastic bulk size) results:")
    print(dtaStochasticResults)
    
    # using hybrid
    print("Running hybrid approach for deterministic bulk size.")
    hybridResults = hybrid.hybrid_DTA_and_fluid(initialWIP, arrivalRateSeries, bulksizeSeries, serviceRate, dta_numberOfStepsPerUnit, dta_K)
    print("Hybrid approach (deterministic bulk size) results:")
    print(hybridResults)
    
    # using simulation
    print("Running simulation for deterministic bulk size.")
    simulationDeterministicResults = simulation.FastSimulationStochasticBatch(initialWIP, arrivalRateSeries, bulksizeSeries, [0]*10, serviceRate,
                                                                                simulation_numberOfIterations, simulation_numberOfSamplesPerUnit)
    print("Simulation (deterministic bulk size) results:")
    print(simulationDeterministicResults)
    
    print("Running simulation for stochastic bulk size.")
    simulationDeterministicResults = simulation.FastSimulationStochasticBatch(initialWIP, arrivalRateSeries, bulksizeSeries, rangeOfBulkSizeSeries, serviceRate,
                                                                                simulation_numberOfIterations, simulation_numberOfSamplesPerUnit)
    print("Simulation (stochastic bulk size) results:")
    print(simulationDeterministicResults)
    
    # getting idle rate from the dta
    print("Running dta for deterministic bulk size idle rate.")
    dtaDeterministicIdleRateResults = hybrid.DTA_deterministic_bulk_size_idletime(initialWIP, arrivalRateSeries, bulksizeSeries, serviceRate, dta_numberOfStepsPerUnit, dta_K)
    print("DTA (deterministic bulk size) idle rate results:")
    print(dtaDeterministicIdleRateResults)
    
    # getting idle rate from the simulation
    print("Running simulation for deterministic bulk size idle rate.")
    simulationDeterministicIdleRateResults = simulation.FastSimulationStochasticBatchIdleRate(initialWIP, arrivalRateSeries, bulksizeSeries, [0]*10, serviceRate,
                                                                                simulation_numberOfIterations, simulation_numberOfSamplesPerUnit)
    print("Simulation (deterministic bulk size)idle rate results:")
    print(simulationDeterministicIdleRateResults)
    
