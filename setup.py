import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'data_reduction'
AUTHOR = 'Paulina Trofimiak'
AUTHOR_EMAIL = 'PaulinaTrofimiak@gmail.com'
URL = 'https://github.com/paultro708/DataReduction'

LICENSE = 'MIT'
DESCRIPTION = 'Package for data reduction, especially using instance selection algorithms.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'matplotlib',
      'scikit-learn',
      'scipy',
      'pytest'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages(),
      python_requires='>=3.6, <3.8'
      )