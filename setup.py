import io
import os

from setuptools import setup, find_packages

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_requirements(filename):
    """Load requirements list."""
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()


def get_long_description():
    """Read README file as a package description."""
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with io.open(readme_path, encoding="utf-8") as f:
        return f"\n{f.read()}"


def get_version():
    """Read package version from the __version__.py file."""
    context = {}
    with open(os.path.join(PROJECT_ROOT, "spifpy", "__version__.py")) as f:
        exec(f.read(), context)
    return context["__version__"]


setup(
    name="spifpy",
    version=get_version(),
    packages=find_packages(),
    description="Single Particle Image Format (SPIF) data converter and interface",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Kenny Bala",
    author_email="Kliti.Bala@nrc-cnrc.gc.ca",
    url="https://github.com/mfreer/SPIFpy",
    install_requires=get_requirements("requirements/requirements.txt"),
    license="MIT",
    keywords="SPIF, Converter",
    classifiers=[
        "Development Status :: Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    package_data={
        "spifpy": ["config/**.ini"],
    },
    entry_points='''
        [console_scripts]
        spifpy=spifpy.scripts.spifpy:spifpy
        spifaddaux=spifpy.scripts.spifaddaux:spifaddaux
        spifcc=spifpy.scripts.spifcc:spifcc
    ''',
    python_requires=">=3.6.0",
)


