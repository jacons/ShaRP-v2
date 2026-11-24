from pathlib import Path
from setuptools import find_packages, setup

with open("sharp/_version.py") as f:
    exec(f.read())

MAINTAINER = "J. Fonseca"
MAINTAINER_EMAIL = "jpfonseca@novaims.unl.pt"
URL = "https://github.com/DataResponsibly/sharp"
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

setup(
    name="xai-sharp",
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    download_url=URL,
    version=__version__,  # noqa: F821
    description="Implementation of the ShaRP framework.",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf8"),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.20.0", "scikit-learn>=1.2.0", "tqdm>=4.46.0"],
)
