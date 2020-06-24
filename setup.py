import setuptools
import glob


with open("README.md", "r") as fh:
    long_description = fh.read()

scripts = glob.glob('bin/*.py')

setuptools.setup(
    name="nireact",
    version="1.0.0",
    author="Neal Morton",
    author_email="mortonne@gmail.com",
    description="Code for analysis of Morton, Schlichting, & Preston (2020)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mortonne/nireact",
    packages=setuptools.find_packages(),
    scripts=scripts,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
