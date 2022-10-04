"""Setups the project."""
import itertools
import re

from setuptools import find_packages, setup

with open("gym/version.py") as file:
    full_version = file.read()
    assert (
        re.match(r'VERSION = "\d\.\d+\.\d+"\n', full_version).group(0) == full_version
    ), f"Unexpected version: {full_version}"
    VERSION = re.search(r"\d\.\d+\.\d+", full_version).group(0)

# Environment-specific dependencies.
extras = {
    "atari": ["ale-py~=0.8.0"],
    "accept-rom-license": ["autorom[accept-rom-license]~=0.4.2"],
    "box2d": ["box2d-py==2.3.5", "pygame==2.1.0", "swig==4.*"],
    "classic_control": ["pygame==2.1.0"],
    "mujoco_py": ["mujoco_py<2.2,>=2.1"],
    "mujoco": ["mujoco==2.2", "imageio>=2.14.1"],
    "toy_text": ["pygame==2.1.0"],
    "other": ["lz4>=3.1.0", "opencv-python>=3.0", "matplotlib>=3.0", "moviepy>=1.0.0"],
}

# Testing dependency groups.
testing_group = set(extras.keys()) - {"accept-rom-license", "atari"}
extras["testing"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], testing_group)))
) + ["pytest==7.0.1"]

# All dependency groups - accept rom license as requires user to run
all_groups = set(extras.keys()) - {"accept-rom-license"}
extras["all"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], all_groups)))
)

# Uses the readme as the description on PyPI
with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

setup(
    author="Gym Community",
    author_email="jkterry@umd.edu",
    classifiers=[
        # Python 3.6 is minimally supported (only with basic gym environments and API)
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Gym: A universal API for reinforcement learning environments",
    extras_require=extras,
    install_requires=[
        "numpy >= 1.18.0",
        "cloudpickle >= 1.2.0",
        "importlib_metadata >= 4.8.0; python_version < '3.10'",
        "gym_notices >= 0.0.4",
        "dataclasses == 0.8; python_version == '3.6'",
    ],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="gym",
    packages=[package for package in find_packages() if package.startswith("gym")],
    package_data={
        "gym": [
            "envs/mujoco/assets/*.xml",
            "envs/classic_control/assets/*.png",
            "envs/toy_text/font/*.ttf",
            "envs/toy_text/img/*.png",
            "py.typed",
        ]
    },
    python_requires=">=3.6",
    tests_require=extras["testing"],
    url="https://www.gymlibrary.dev/",
    version=VERSION,
    zip_safe=False,
)
