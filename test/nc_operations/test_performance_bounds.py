"""Test of the performance bounds."""

import pytest

from nc_arrivals.qt import DM1
from nc_operations.performance_bounds import backlog_prob, delay_prob, output
from nc_server.constant_rate_server import ConstantRateServer


def test_backlog_prob():
    assert backlog_prob(arr=DM1(lamb=1.2),
                        ser=ConstantRateServer(2.0),
                        theta=1.0,
                        backlog_value=3.0,
                        indep=True) == pytest.approx(0.2648413131)

    assert backlog_prob(arr=DM1(lamb=1.2),
                        ser=ConstantRateServer(3.0),
                        theta=1.0,
                        backlog_value=3.0,
                        indep=True) == pytest.approx(0.07099480875)

    assert backlog_prob(arr=DM1(lamb=1.2),
                        ser=ConstantRateServer(3.0),
                        theta=0.5,
                        backlog_value=3.0,
                        indep=False,
                        p=2.0) == pytest.approx(0.4920777142)

    assert backlog_prob(arr=DM1(lamb=1.0),
                        ser=ConstantRateServer(1.6),
                        theta=0.6,
                        backlog_value=10.0,
                        indep=True) == pytest.approx(0.05795839492)

    assert backlog_prob(arr=DM1(lamb=1.2),
                        ser=ConstantRateServer(1.2),
                        theta=0.5,
                        backlog_value=10.0,
                        indep=True) == pytest.approx(0.113855036)


def test_delay_prob():
    assert delay_prob(arr=DM1(lamb=1.2),
                      ser=ConstantRateServer(2.0),
                      theta=1.0,
                      delay_value=3,
                      indep=True) == pytest.approx(0.01318567256)

    assert delay_prob(arr=DM1(lamb=1.2),
                      ser=ConstantRateServer(3.0),
                      theta=1.0,
                      delay_value=3,
                      indep=True) == pytest.approx(0.0001759785367)

    assert delay_prob(arr=DM1(lamb=1.2),
                      ser=ConstantRateServer(3.0),
                      theta=0.5,
                      delay_value=3,
                      indep=False,
                      p=2.0) == pytest.approx(0.0244991068)

    assert delay_prob(arr=DM1(lamb=1.0),
                      ser=ConstantRateServer(1.6),
                      theta=0.6,
                      delay_value=10,
                      indep=True) == pytest.approx(0.001583639096)

    assert delay_prob(arr=DM1(lamb=1.2),
                      ser=ConstantRateServer(1.2),
                      theta=0.5,
                      delay_value=10,
                      indep=True) == pytest.approx(0.04188492703)


def test_output():
    assert output(arr=DM1(lamb=1.2),
                  ser=ConstantRateServer(2.0),
                  theta=1.0,
                  delta_time=3,
                  indep=True) == pytest.approx(1149.007674)

    assert output(arr=DM1(lamb=1.2),
                  ser=ConstantRateServer(3.0),
                  theta=1.0,
                  delta_time=3,
                  indep=True) == pytest.approx(308.0092721)

    assert output(arr=DM1(lamb=1.2),
                  ser=ConstantRateServer(3.0),
                  theta=0.5,
                  delta_time=3,
                  indep=False,
                  p=2.0) == pytest.approx(32.41173617)
