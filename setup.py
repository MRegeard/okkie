from setuptools import find_packages, setup

setup(
    name="okkie",
    version="0.0.1",
    packages=find_packages(
        include=[
            "astropy",
            "gammapy>=1.2",
            "matplotlib>=3.4",
            "scipy!=1.10",
            "iminuit>=2.8.0",
            "regions>=0.5",
            "numpy>1.20",
            "pint-pulsar~=0.9.3",
            "ruff",
        ]
    ),
)
