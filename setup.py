
from setuptools import setup


setup(
    name='Diffusion_Model',
    version='1.1',
    description='A python module to implement a Simple Diffusion Model',
    author='Sajjad Mahdavi',
    packages=[
        "data",
        "models",
        "train"
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'torch', 
        'timm',
        'einops',
        'tensorboard'
    ],
)
