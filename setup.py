from setuptools import setup, find_packages

setup(
    name='IBegToDiffer',
    version='0.01',
    author="Brian Walshe",
    packages=find_packages(),
    license='Apache 2.0',
    description="Some kind of autodiff thing",
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    extras_require={
        "test": ['pytest']
    }
)