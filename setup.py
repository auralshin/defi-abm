# setup.py

from setuptools import setup, find_packages

setup(
    name="defi-abm",
    version="1.0.0",
    description="Agent-based DeFi simulation plugin for Mesa",
    author="Your Name",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mesa>=1.0.0",
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.0.0",
        "PyYAML>=5.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
