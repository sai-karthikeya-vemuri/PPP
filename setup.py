from setuptools import setup

setup(name='autodiff',
      version='0.1',
      description='Rudimentary automatic differentiation framework',

      packages=['autodiff', 'autodiff.core'],
      license='MIT',
      install_requires=['numpy', 'graphviz'],
      )
