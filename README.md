# Python MGF Calculator

The (sigma, rho)-calculus is implemented in this "tool."
It is supposed to provide quick solutions for small networks.
Yet, it must not show weak points when it comes to parameter optimization.
The new approach using Lyapunov's inequality for the output is also included.

## Prerequisites

- Python 3.6.5


## Introduction

#### Compute Bound Directly

The easiest start is to use `Calculate_Examples.py`.
Set a performance parameter, e.g., by
```python
OUTPUT_TIME6 = PerformParameter(perform_metric=PerformMetric.OUTPUT, value=6)
```
This means that we want the compute the output for time delta = 6.
```python
SINGLE_SERVER = SingleServerPerform(arr=ExponentialArrival(lamb=1.0),
                                    ser=ConstantRate(rate=10.0),
                                    perform_param=OUTPUT_TIME6)
```
means that we consider the single hop topology we exponentially distributed arrival (parameter lambda equal to 1) and a constant rate server (with rate 10).

```python
print(SINGLE_SERVER.get_bound(0.1))
```
gives then the bound at theta = 0.1.

For the new output bound computation, we have to set a list of parameters:
```python
[0.1, 2.7]
```
and compute
```python
print(SINGLE_SERVER.get_new_bound([0.1, 2.7]))
```

#### Compute Optimized Bound
Assume we want to optimize the parameter for the above setting.
Therefore, we choose to optimize via grid search.
We optimize the bound of the SINGLE_SERVER setting with the old approach "Optimize" and the new approach OptimizeNew and want to print the optimal parameter set ("print_x=true"). The last step is to choose the method "grid_search()" and to set the granularity (in this case = 0.1):
```python
print(Optimize(SINGLE_SERVER, print_x=True).grid_search(
        bound_list=[(0.1, 5.0)], delta=0.1))
```
and for the new version:
```python
print(OptimizeNew(SINGLE_SERVER, new=True, print_x=True).grid_search(
        bound_list=[(0.1, 5.0), (0.9, 8.0)], delta=0.1))
```


## Status of Implementation

Network Calculus operations:
- Convolution
- Deconvolution
- Aggregation
- Leftover service

Performance Metrics:
- Backlog (probability)
- Delay (probability)
- Output

Arrival processes:
- Exponential Distribution
- MMOO
- Token Bucket with constant parameters
- Leaky Bucket after application of the conversion theorem to the Massoulié results

For the service, only a constant rate server is available.

Topologies / settings:
- Single server
- Fat tree

## Folder Structure

- dnc  
Only contains the delay bound for deterministic token bucket arrivals
- library  
Contains all helper classes, for example:
  - `array_to_results.py` takes an input Numpy-array, performs an analysis and write the results in a dictionary
  - `compare_old_new.py` compares the standard approach with the new bound
  - `perform_parameter.py` stores emum PerformMetric (delay, output,...) and its value
-  nc_operations  
Network Calculus Operations, (De-)Convolution and computation of performance metrics (delay, backlog, delay probability)
- nc_processes
All arrival and service processes, such as MMOO and constant rate server
- optimization  
All classes that are necessary for the parameter optimization, as this is an important aspect of MGF-calculus.
- simulation and simulation fluid  
Simulation, i.e., no bound computation, with packets or in a fluid model. Only very basic.

The topologies "single server" and "fat tree" have their own dedicated folders as they also include all classes needed for a performance evaluation (random input parameters, save results in csv-files)
