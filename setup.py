from setuptools import setup, find_packages

setup(name='example5',
  version='0.1',
  packages=find_packages(),
  description='run keras on icloud',
  author='Jeffrey Chen',
  author_email='jchen623@stanford.edu',
  license='MIT',
  install_requires=[
      'keras',
      'h5py'
  ],
  zip_safe=False)
