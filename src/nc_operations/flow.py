"""Flow class that incorporates arrival and server information"""

from typing import List
from nc_arrivals.arrival_distribution import ArrivalDistribution


class Flow(object):
    def __init__(self, arr: ArrivalDistribution, server_indices: List[int]):
        self.arr = arr
        self.server_indices = server_indices
