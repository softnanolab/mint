from setuptools import find_packages, setup

with open("mint/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()

extras = {
    "mint": [
        "biopython",
        "deepspeed==0.5.9",
        "dm-tree",
        "lightning==1.9.5",
        "omegaconf",
        "einops",
        "scipy",
    ]
}

setup(
    name="mint",
    version=version,
    description="Learning the language of protein-protein interactions",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["mint", "mint.*"]),
    author="Varun Ullanat",
    url="https://github.com/VarunUllanat/mint",
    license="MIT",
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
    zip_safe=False,
)
