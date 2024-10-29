#!/usr/bin/env python

"""
 Copyright (C) 2023 Università degli Studi di Camerino.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta, Luca Mozzoni, Vincenzo Nucci

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

"""The setup script."""

import versioneer

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("/opt/atlassian/pipelines/agent/build/requirements.txt") as f:
    requirements = f.read().splitlines()

with open("/opt/atlassian/pipelines/agent/build/requirements_dev.txt") as f:
    requirements_full = f.read().splitlines()

extras_require = {
    "full": requirements_full
}

test_requirements = [
    "pytest>=3",
]

setup(
    author="Università degli Studi di Camerino",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        # "LICENSE :: OSI APPROVED :: GNU General Public License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Rain library.",
    install_requires=requirements,
    extras_require=extras_require,
    license="GNU General Public License",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="rain",
    name="rain-dm",
    packages=find_packages(include=["rain", "rain.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://bitbucket.org/proslabteam/rain_unicam",
    version="1." + versioneer.get_increasing_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
