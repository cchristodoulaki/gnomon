from setuptools import setup, find_packages

setup(
    name='gnomon',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    author='Christina Christodoulakis',
    author_email='christina@cs.toronto.edu',
    url='https://github.com/cchristodoulaki/gnomon',
    python_requires='==3.8.10'
)