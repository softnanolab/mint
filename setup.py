from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

# Minimal setup.py - most configuration is now in pyproject.toml
setup(
    name="mint",
    version="0.1.0",
    description="Learning the language of protein-protein interactions",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author="Varun Ullanat",
    url="https://github.com/VarunUllanat/mint",
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    include_package_data=True,
    zip_safe=False,
)
