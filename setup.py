"""Setups the project."""
import itertools
import re

from setuptools import find_packages, setup

with open('gym/version.py') as file:
    full_version = file.read()
    assert re.match(r'VERSION = "\d\.\d+\.\d+"\n', full_version).group(0) == full_version, \
        f'Unexpected version re'
    VERSION = re.search(r'\d\.\d+\.\d+', full_version).group(0)

# Environment-specific dependencies.
extras = {
    "atari": ["ale-py~=0.7.5"],
    "accept-rom-license": ["autorom[accept-rom-license]~=0.4.2"],
    "box2d": ["box2d-py==2.3.5", "pygame==2.1.0"],
    "classic_control": ["pygame==2.1.0"],
    "mujoco_py": ["mujoco_py<2.2,>=2.1"],
    "mujoco": ["mujoco==2.2.0", "imageio>=2.14.1"],
    "toy_text": ["pygame==2.1.0"],
    "other": ["lz4>=3.1.0", "opencv-python>=3.0", "matplotlib>=3.0"],
}

# Testing dependency groups.
testing_group = set(extras.keys()) - {"accept-rom-license", "atari"}
extras["testing"] = list(set(
    itertools.chain.from_iterable(map(lambda group: extras[group], testing_group))
)) + ["pytest", "mock"]

# All dependency groups
all_groups = set(extras.keys()) - {"accept-rom-license"}
extras["all"] = list(set(
    itertools.chain.from_iterable(map(lambda group: extras[group], all_groups))
))

# Gets the requirements from "requirements.txt"
with open('requirements.txt') as file:
    install_requirements = list(map(lambda line: line.strip(), file.readlines()))

# Updates the test_requirements.txt based on `extras["testing"]`
with open('test_requirements.txt', 'w') as file:
    file.writelines(list(map(lambda line: f'{line}\n', extras["testing"])))

setup(
    name="gym",
    version=VERSION,
    description="Gym: A universal API for reinforcement learning environments",
    url="https://www.gymlibrary.ml/",
    author="Gym Community",
    author_email="jkterry@umd.edu",
    license="MIT",
    packages=[package for package in find_packages() if package.startswith("gym")],
    zip_safe=False,
    install_requires=install_requirements,
    extras_require=extras,
    tests_require=extras["testing"],
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
