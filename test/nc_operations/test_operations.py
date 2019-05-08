"""Test of the network operations."""

import pytest

from nc_arrivals.qt import DM1
from nc_operations.operations import Convolve, Deconvolve, Leftover
from nc_server.constant_rate_server import ConstantRateServer


def test_deconvolve_sigma():
    assert Deconvolve(
        arr=DM1(lamb=1.2), ser=ConstantRateServer(2.0),
        indep=True).sigma(theta=1.0) == pytest.approx(1.671375549)


def test_deconvolve_rho():
    assert Deconvolve(
        arr=DM1(lamb=1.2), ser=ConstantRateServer(2.0),
        indep=True).rho(theta=1.0) == pytest.approx(1.791759469)


def test_leftover_sigma():
    assert Leftover(
        ser=ConstantRateServer(2.0), arr=DM1(lamb=1.2), indep=False,
        p=2.0).sigma(theta=0.5) == pytest.approx(0.0)


def test_leftover_rho():
    assert Leftover(
        ser=ConstantRateServer(2.0), arr=DM1(lamb=1.2), indep=True,
        p=1.0).rho(theta=0.5) == pytest.approx(0.9220069985)

    assert Leftover(
        ser=ConstantRateServer(2.0), arr=DM1(lamb=1.2), indep=False,
        p=2.0).rho(theta=0.5) == pytest.approx(0.2082405308)

    assert Leftover(
        ser=ConstantRateServer(2.0), arr=DM1(lamb=1.2), indep=False,
        p=1.1).rho(theta=0.5) == pytest.approx(0.8852645948)


def test_convolve_leftover_sigma():
    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(
            ser=ConstantRateServer(rate=3.0),
            arr=DM1(lamb=1.2))).sigma(theta=1.0) == pytest.approx(0.0)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=2.0).sigma(theta=0.5) == pytest.approx(0.0)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=1.8).sigma(theta=0.5) == pytest.approx(1.988291179)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(
            ser=ConstantRateServer(rate=4.0),
            arr=DM1(lamb=1.2))).sigma(theta=1.0) == pytest.approx(0.4586751454)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=4.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=2.0).sigma(theta=0.5) == pytest.approx(1.865504259)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=4.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=1.8).sigma(theta=0.5) == pytest.approx(6.583291316)


def test_convolve_leftover_rho():
    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(
            ser=ConstantRateServer(rate=3.0),
            arr=DM1(lamb=1.2))).rho(theta=1.0) == pytest.approx(0.2082405308)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=2.0).rho(theta=0.5) == pytest.approx(-0.7917594692)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=1.8).rho(theta=0.5) == pytest.approx(0.5354766913)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(
            ser=ConstantRateServer(rate=4.0),
            arr=DM1(lamb=1.2))).rho(theta=1.0) == pytest.approx(1.208240531)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=4.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=2.0).rho(theta=0.5) == pytest.approx(1.208240531)

    assert Convolve(
        ser1=Leftover(ser=ConstantRateServer(rate=3.0), arr=DM1(lamb=1.2)),
        ser2=Leftover(ser=ConstantRateServer(rate=4.0), arr=DM1(lamb=1.2)),
        indep=False,
        p=1.8).rho(theta=0.5) == pytest.approx(1.459672932)
