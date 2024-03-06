from setuptools import find_packages, setup

DESCRIPTION = (
    "sreft-ml: Related toolset for building disease progression models by sreft-ml"
)
NAME = "sreft-ml"
AUTHOR = "Ryota Jin"
AUTHOR_EMAIL = "riu2309j@gmail.com"
URL = "https://github.com/RyotaJin/sreft-ml.git"
LICENSE = "MIT"
DOWNLOAD_URL = "https://github.com/RyotaJin/sreft-ml.git"
VERSION = "0.2.0"
PYTHON_REQUIRES = ">=3.10"

INSTALL_REQUIRES = [
    "lifelines>=0.28.0",
    "matplotlib>=3.7.1",
    "numpy>=1.23.5",
    "pandas>=2.0.3",
    "scikit-learn>=1.2.2",
    "scipy>=1.11.2",
    "seaborn>=0.12.2",
    "shap>=0.44.1",
    "statsmodels>=0.14.0",
    "tensorflow>=2.10",
]

PACKAGES = find_packages()

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

with open("README.md", "r") as fp:
    readme = fp.read()
long_description = readme

setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
)
