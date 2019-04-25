from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
        'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(
    name='rl_botics',
    version='0.1.0',
    author='Suman Pal',
    author_email='suman7495@gmail.com',
    url='https://github.com/Suman7495/rl-botics.git',
    description='Deep Reinforcement Learning Toolbox for Robotics',
    packages=[package for package in find_packages()
              if package.startswith('rl_botics')],
    license='LICENSE',
    install_requires=[
            'gym',
            'numpy',
            'scipy',
            'tensorflow',
            'tensorflow_probability',
            'pandas',
            'keras',
            'matplotlib'
    ]
)
