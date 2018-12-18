from codecs import open
from os import path

from setuptools import setup, find_packages

# adding README
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neuralkernel',
    version='0.0.8',
    description='neural networks as a general-purpose computational framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Noah Stebbins',
    author_email='nstebbins1@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy', 'matplotlib', 'seaborn'],
    tests_require=['pytest']
)
