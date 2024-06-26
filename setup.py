import os
import setuptools
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.mf file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name="lily", # Replace with your own username
    version="0.0.1",
    author="Weiran Yao",
    author_email="lily@googlegroups.com",
    description="The LiLY is a tool for discovering latent temporal causal factors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weirayao/delta",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)