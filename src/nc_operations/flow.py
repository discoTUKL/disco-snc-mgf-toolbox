"""Flow class that incorporates arrival and server information"""

from typing import List
from nc_arrivals.arrival import Arrival


class Flow(object):
    def __init__(self, arr: Arrival, server_indices: List[int]):
        self.arr = arr
        self.server_indices = server_indices

    def __str__(self):
        return str(self.arr) + "_server_indices=" + str(self.server_indices)

    def __repr__(self):
        return str(self)
