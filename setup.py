from setuptools import find_packages, setup

setup(
    name='bank-marketing-prediction',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'run_prediction=src.main:main',
        ],
    },
)