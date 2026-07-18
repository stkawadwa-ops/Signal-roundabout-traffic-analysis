from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="signal-roundabout-traffic-analysis",
    version="0.1.0",
    author="stkawadwa-ops",
    description="Sophisticated vehicle detection and traffic flow analysis at signalized roundabouts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stkawadwa-ops/Signal-roundabout-traffic-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
)
