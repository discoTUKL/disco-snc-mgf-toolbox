"""IID Arrivals"""

from abc import abstractmethod

from nc_arrivals.arrival_distribution import ArrivalDistribution


class IIDArrivals(ArrivalDistribution):
    """Abstract class for arrival processes that are iid."""

    def sigma(self, theta=0.0) -> float:
        """

        :param theta: mgf parameter
        :return:      sigma(theta)
        """
        return 0.0

    @abstractmethod
    def rho(self, theta: float) -> float:
        """
        rho(theta)
        :param theta: mgf parameter
        """
        pass

    def is_discrete(self) -> bool:
        """
        :return True if the arrival distribution is discrete, False if not
        """
        return True

    @abstractmethod
    def average_rate(self) -> float:
        pass

    @abstractmethod
    def to_value(self, number=1, show_n=False) -> str:
        pass
