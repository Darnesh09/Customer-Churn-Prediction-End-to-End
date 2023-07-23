from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
        This file reads the requirements from file_path
    """

    requirements = []
    with open(file_path) as file:
        requirements = [req.strip() for req in file.readlines()]
    if '-e.' in requirements:
        requirements.remove('-e.')
    return requirements

setup(
    name = "Customer Churn Prediction",
    version = '0.0.1',
    author='Darnesh',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)