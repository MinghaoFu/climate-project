from setuptools import setup, find_packages

setup(
    name='Caulimate',
    version='0.1.3',
    author='Minghao Fu',
    author_email='isminghaofu@gmail.com',
    description='Causality on Climate Science',
    long_description=open('README.md').read(),
    url='https://github.com/MinghaoFu/Caulimate',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
