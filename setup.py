from setuptools import setup, find_packages

setup(
    name='neuralkernel',
    version='0.0.1',
    description='neural networks as a general-purpose computational framework',
    author='Noah Stebbins',
    author_email='nstebbins1@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy', 'matplotlib'],
)
