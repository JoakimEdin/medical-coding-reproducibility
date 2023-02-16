from setuptools import find_packages, setup


# Get requirements file depending availability of CUDA enabled GPU on the system and whether to use slim requirements
requirements_file = "requirements.txt"

# Read requirements file
with open(requirements_file) as f:
    requirements = f.read().splitlines()
print(
    "Found the following requirements to be installed from {}:\n  {}".format(
        requirements_file, "\n  ".join(requirements)
    )
)

# Collect packages
packages = find_packages(exclude=("tests", "experiments"))
print("Found the following packages to be created:\n  {}".format("\n  ".join(packages)))

# Get long description from README
with open("README.md", "r") as readme:
    long_description = readme.read()

# Setup the package
setup(
    name="automatic-medical-coding",
    version="1.0.0",
    packages=packages,
    python_requires=">=3.10.0",
    install_requires=requirements,
    setup_requires=[],
    ext_modules=[],
    url="",
    author="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
