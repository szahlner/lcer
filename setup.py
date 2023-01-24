import os

from setuptools import setup

with open(os.path.join("lcer", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

long_description = """
# Local Cluster Experience Replay (LCER)
LCER is an application for replay-buffers to increase the sample efficiency of off-policy reinforcement learning algorithms. The goal is to reduce the number of required interactions with the environment to maximize the per step reward and, in turn, the cumulative reward. At the same time, the additional overhead in terms of implementation effort and computational power should be kept at a minimum.
See https://github.com/szahlner/lcer
"""

setup(
    name="lcer",
    packages=["lcer"],
    package_data={
        "lcer": [
            "version.txt",
        ]
    },
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
        "gym",
        "mpi4py",
        "pytest",
    ],
    description="Local Cluster Experience Replay (LCER)",
    author="Stefan Zahlner",
    url="https://github.com/szahlner/lcer",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.7",
)
