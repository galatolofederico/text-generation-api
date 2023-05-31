import setuptools
import os

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text-generation-api",
    version="0.0.1",
    author="Federico A. Galatolo",
    author_email="federico.galatolo@unipi.it",
    description="",
    url="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    entry_points = {
        "console_scripts": [
            "text-generation-api=text_generation_api.cmd:main",
        ],
    },
    install_requires=[
        "transformers==4.29.2",
        "PyYAML==6.0",
        "fastapi==0.95.2",
        "uvicorn==0.22.0",
        "httpx==0.24.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
)