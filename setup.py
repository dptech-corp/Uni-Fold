# Copyright 2022 DP Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="unifold",
    version="2.2.1",
    description="An open-source platform for developing protein folding models beyond AlphaFold.",
    author="DP Technology",
    author_email="unifold@dp.tech",
    license="Apache License, Version 2.0",
    url="https://github.com/dptech-corp/Uni-Fold",
    packages=find_packages(
        exclude=["scripts", "tests", "example_data", "docker", "benchmark", "img", "evaluation", "notebooks"]
    ),
    install_requires=[
        "absl-py",
        "biopython",
        "ml-collections",
        "numpy",
        "pandas",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
