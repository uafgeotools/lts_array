from setuptools import setup, find_packages
import os as os

# https://github.com/readthedocs/readthedocs.org/issues/5512#issuecomment-475073310
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['matplotlib', 'numpy', 'obspy', 'scipy', 'numba']

setup(
   name='lts_array',
   version='2.0',
   description='Apply least trimmed squares to infrasound and seismic array processing.',
   license='LICENSE.txt',
   author='Jordan W. Bishop',
   url="https://github.com/uafgeotools/lts_array",
   packages=find_packages(),
   python_requires='>=3.0',
   install_requires=INSTALL_REQUIRES,
   scripts=[
            'example.py',
           ]
)
