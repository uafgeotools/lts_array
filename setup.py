from setuptools import setup, find_packages
import os as os

# https://github.com/readthedocs/readthedocs.org/issues/5512#issuecomment-475073310
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['matplotlib', 'numpy', 'obspy', 'scipy']

setup(
   name='lts_array',
   version='1.1',
   description='Apply least trimmed squares to infra/seis array processing.',
   license='LICENSE.txt',
   author='Jordan W. Bishop',
   url="https://github.com/uafgeotools/lts_array",
   packages=find_packages(),
   python_requires='>=3.0',
   install_requires=INSTALL_REQUIRES,
   scripts=[
            'Example_Processing.py',
           ]
)
