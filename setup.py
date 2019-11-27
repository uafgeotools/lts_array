from setuptools import setup, find_packages

setup(
   name='lts_array',
   version='1.1',
   description='Apply least trimmed squares to infra/seis array processing.',
   license='LICENSE.txt',
   author='Jordan W. Bishop',
   url="https://github.com/uafgeotools/lts_array",
   packages=find_packages(),
   python_requires='>=3.0',
   install_requires=['numpy', 'scipy', 'obspy', 'copy'],
   scripts=[
            'Example_Processing.py',
           ]
)
