# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
NAME = 'dmtools'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get version
with open(path.join(here, NAME, 'VERSION'), encoding='utf-8') as f:
    version = f.read()

setup(
    name=NAME,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description="Python DM tools for MagAO-X",
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/magao-x/dmtools',

    # Author details
    author='Kyle Van Gorkom',
    author_email='kvangorkom@email.arizona.edu',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # add shell scripts here
    entry_points = {
        'console_scripts': ['dm_project_zernikes=dmtools.project_zernikes:main',
                            'dm_offload_matrix=dmtools.woofer_tweeter_offload:main',
                            'dm_eye_doctor=dmtools.eye_doctor:main'],
    },
    
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy','astropy', 'poppy'],
)