from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='pydecomposer',
    version='1.1.3',
    packages=find_packages(),
    install_requires=requirements,
    description='A tool for signal decomposition',
    author='Yassine Charouif',
    author_email='yc4923@ic.ac.uk'
)