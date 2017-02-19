from __future__ import with_statement
import os
from setuptools import find_packages


## windows install part from http://matthew-brett.github.io/pydagogue/installing_scripts.html
import os
from os.path import join as pjoin, splitext, split as psplit
from distutils.core import setup
from distutils.command.install_scripts import install_scripts
from distutils import log


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "DSST",
    version = "0.1.0",
    author = "Peter Rennert",
    author_email = "p.rennert@cs.ucl.ac.uk",
    description = ("Python implementation of Discriminative Scale Space Tracker"),
    #license = read('LICENSE.txt'),
    keywords = "tracking",
    url = "https://github.com/groakat/dsst",
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
    ],
    package_data = {
        '': ['*.svg', '*.yaml', '*.zip', '*.ico', '*.bat']
    }
)