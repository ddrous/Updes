from setuptools import setup

setup(
   name='Updes',
   version='1.0.3',
   author='ddrous',
   author_email='desmond.ngueguin@gmail.com',
   packages=['updes', 'updes.tests'],
   url='http://pypi.python.org/pypi/Updes/',
   license='LICENSE.md',
   description='A package for meshless PDE modelling and control',
   long_description=open('README.md', encoding="utf-8").read(),
   install_requires=[
       "scikit-learn",
       "jax >= 0.3.4",
       "pytest",
       "gmsh",
       "matplotlib>=3.4.0",
       "seaborn",
   ],
)
