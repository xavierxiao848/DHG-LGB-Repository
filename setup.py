"""
Setup script for DHG-LGB package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    filepath = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='dhg-lgb',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@institution.edu',
    description='Disease-Hypergraph Integrated with LightGBM for Metabolite-Disease Association Prediction',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/DHG-LGB',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'black>=21.9b0',
            'flake8>=3.9.0',
            'sphinx>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'dhg-lgb-preprocess=scripts.01_preprocess_data:main',
            'dhg-lgb-train-hgnn=scripts.02_train_hgnn:main',
            'dhg-lgb-train-classifier=scripts.03_train_classifier:main',
            'dhg-lgb-evaluate=scripts.04_evaluate_model:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
