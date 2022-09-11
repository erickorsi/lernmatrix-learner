from setuptools import setup

from __init__ import __version__

setup(
    name='lernmatrix-learner',
    version=__version__,

    url='https://github.com/erickorsi/lernmatrix-learner',
    author='Erick Hotta Orsi',
    author_email='erickorsig@gmail.com',
    py_modules=['lernmatrix'],
    install_requires=[
        'numpy',
    ]
)