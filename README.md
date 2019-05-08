# Stochastic Network Calculus with Moment Generating Functions

The (sigma, rho)-calculus is implemented in this tool. It is supposed to provide solutions quickly for small networks. Yet, it must not show weak points when it comes to parameter optimization. It also includes a new approach using Jensen inequality to improve the output bound.

## Prerequisites

- Python 3.7 or higher
- Python packages in `requiremets.txt`

## Introduction

#### Compute Bound Directly

The easiest start is to use `Calculate_Examples.py`. Set a performance parameter, e.g., by

```python
OUTPUT_TIME6 = PerformParameter(perform_metric=PerformMetric.OUTPUT, value=6)
```

we compute the output bound for time delta = 6.

```python
SINGLE_SERVER = SingleServerPerform(arr=DM1(lamb=1.0),
                                    ser=ConstantRate(rate=10.0),
                                    perform_param=OUTPUT_TIME6)
```

means that we consider the single hop topology with exponentially distributed arrival increments (DM1 with parameter lambda equal to 1) and a constant rate server (with rate 10).

```python
print(SINGLE_SERVER.bound(0.1))
```

for theta = 0.1.

For the output bound computation with the $h$ - mitigator, we insert a list of parameters (first element is theta, the rest are for the new output bound computation):

```python
[0.1, 2.7]
```

and compute

```python
print(SINGLE_SERVER.h_mit_bound([0.1, 2.7]))
```

#### Compute Optimized Bound

Assume we want to optimize the parameter for the above setting. Therefore, we choose to optimize, e.g., via grid search.
We optimize the bound in the SINGLE_SERVER setting with the old approach "Optimize" and the approach OptimizeMitigator and want to print the optimal parameter set ("print_x=true"). The last step is to choose the method "grid_search()" and to set the search's granularity (in this case = 0.1):

```python
print(Optimize(SINGLE_SERVER, print_x=True).grid_search(
        bound_list=[(0.1, 5.0)], delta=0.1))
```

and for the new version:

```python
print(OptimizeMitigator(SINGLE_SERVER, new=True, print_x=True).grid_search(
        bound_list=[(0.1, 5.0), (0.9, 8.0)], delta=0.1))
```

## Status of Implementation

Network Calculus operations:

- Convolution
- Deconvolution
- Aggregation
- Leftover service

Performance Metrics:

- Backlog bound for a given probability and vice versa
- Delay bound for a given probability and vice versa
- MGF-output bound

Arrival processes:

- DM1
- MD1
- Markov modulated on-off traffic (MMOO)
- Exponentially bounded burstiness (EBB)
- Token Bucket

For the service, only a constant rate server is available.

Topologies / settings:

- Single server
- Fat tree
- Canonical tandem

## Folder Structure

- nc_arrivals
  All arrival processes, such as MMOO and constant rate server
- nc_operations
  Network Calculus Operations, (De-)Convolution and computation of performance metrics (delay, backlog, delay probability)
- nc_service
  Constant rate server and the abstract class Service
- optimization
  All classes that are necessary for the parameter optimization, as this is an important aspect of MGF-calculus.
- utils
  Contains all helper classes, for example:
  - `exceptions.py` includes special ParameterOutOfBounds class that is handled in the optimization
  - `perform_parameter.py` stores emum PerformMetric (delay, output,...) and its value
- h_mitigator
  contains all classes and functions concerning the new $h$-mitigator approach to improve performance bounds

The topologies have their own dedicated folders as they also include all classes needed for a performance evaluation (random input parameters, save results in csv-files)
