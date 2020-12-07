from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("LICENSE", "r", encoding="utf-8") as f:
    LICENSE = f.read()

setup(
    name='RandomizedResponse',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/tchaye59/RandomizedResponse',
    license=LICENSE,
    author='Jude TCHAYE',
    author_email='tchaye59@gmail.com',
    description='Applies a RandomizedResponse to the input. The RandomizedResponse layer randomly sets input units to '
                'random values with a frequency of rate at each step during training time, which helps prevent '
                'overfitting. ',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
