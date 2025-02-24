from setuptools import setup


with open("esm/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()

extras = {
    "esmfold": [ # OpenFold does not automatically pip install requirements, so we add them here.
        "biopython",
        "deepspeed==0.5.9",
        "dm-tree",
        "lightning==1.9.5",
        "omegaconf",
        "ml-collections",
        "einops",
        "scipy",
    ]
}


setup(
    name="plm-multimer",
    version=version,
    description="Learning the language of protein-protein interactions with PLM-Multimer",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Varun Ullanat",
    url="https://github.com/VarunUllanat/plm-multimer",
    license="MIT",
    extras_require=extras,
    classifiers=[
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
    include_package_data=True,
    zip_safe=False,
)
