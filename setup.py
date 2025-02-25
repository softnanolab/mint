from setuptools import setup, find_packages

with open("plm_multimer/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()

extras = {
    "plm_multimer": [
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
    name="plm_multimer",
    version=version,
    description="Learning the language of protein-protein interactions with PLM-Multimer",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["plm_multimer", "plm_multimer.*"]),
    author="Varun Ullanat",
    url="https://github.com/VarunUllanat/plm-multimer",
    license="MIT",
    extras_require=extras,
    classifiers=[
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.7',
        ],
    include_package_data=True,
    zip_safe=False,
)
