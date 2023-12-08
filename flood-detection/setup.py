from setuptools import setup, find_packages

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    package_data={'trainer': ['configs/*.json']},
    install_requires=[
        'numpy>=1.18.0',
        'tensorflow>=2.12.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.0.0',
        'google-cloud-logging',
        'google-cloud-storage'
    ],
)