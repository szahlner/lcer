import os

from setuptools import setup

with open(os.path.join("lcer", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

long_description = """
# RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents
See https://github.com/DLR-RM/rl-baselines3-zoo
"""

setup(
    name="lcer",
    packages=["lcer"],
    package_data={
        "lcer": [
            "version.txt",
        ]
    },
    # entry_points={"console_scripts": ["rl_zoo3=rl_zoo3.cli:main"]},
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
        "gym",
        "mpi4py",
        "pytest",
    ],
    # extras_require={
    #     "plots": ["seaborn", "rliable>=1.0.5", "scipy~=1.7.3"],
    # },
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
