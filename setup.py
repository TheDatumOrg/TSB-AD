from setuptools import setup, find_packages
from os import path
from io import open

this_directory = path.abspath(path.dirname(__file__))

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='TSB_AD',  # Replace with your own package name
    version='1.0',  # The version of your package
    author='The Datum Lab',  # Your name
    description='Time-Series Anomaly Detection Benchmark',  # A short description
    long_description=open('README.md', encoding='utf-8').read(),  # Long description read from the README.md
    long_description_content_type='text/markdown',  # Type of the long description, typically text/markdown or text/x-rst
    url='https://github.com/TheDatumOrg/TSB-AD',  # Link to the repository or website
    packages=find_packages(),  # List of all Python import packages that should be included in the Distribution Package
    install_requires=requirements,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9'
        'Programming Language :: Python :: 3.10'
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',  # Minimum version requirement of the package
    entry_points={},
    include_package_data=True,  # Whether to include non-code files in the package
    zip_safe=False,  # Whether the package can be run out of a zip file
)