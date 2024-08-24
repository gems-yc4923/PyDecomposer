from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='pydecomposer',
    version='1.1.5',  # Incrementing the version number
    packages=find_packages(exclude=['tests*']),
    install_requires=requirements,
    description='A tool for advanced signal decomposition using VMD and CEEMDAN',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yassine Charouif',
    author_email='yc4923@ic.ac.uk',
    url='https://github.com/gems-yc4923/PyDecomposer',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    keywords='signal decomposition vmd ceemdan',
)