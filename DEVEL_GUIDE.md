# Development guide

To get started, first, get your local repository:

```bash
# Clone the repository
git clone <repository-url> <folder-name>
cd <folder-name>

# Create an appropriate environment
conda env create -f environment.yml
conda activate rco_devel

# Local installation in editable mode to reflect changes made in the code
# Note: Editable installs are done within the conda environment using pip
pip install -e .  
```

After such installation, you can edit the code and then use `rc-optimizer` from anywhere to test your developments.

## Python Package Configuration
All package configuration is in `pyproject.toml`.

## Conda Package
The Conda package recipe is in [conda-recipe/meta.yaml](conda-recipe/meta.yaml). Using this recipe, the github workflow defined at [.github/workflows/publish_conda.yml](.github/workflows/publish_conda.yml) publishes a new version for every version tag adhering to the semantic versioning syntax `vMajor.Minor.Patch` (where `Major`, `Minor`, and `Patch` are integers).

If you need to test the Conda package, for example when changing something in the conda recipe, you can build it locally:

```bash
cd conda-recipe

# Can use either conda or mamba (faster)
conda env create -f environment.yml
conda activate rco_package_builder

# Here we set the flag to avoid publishing the locally built package
conda config --set anaconda_upload no

# Build the Package
conda build .
```

This will take some minutes. Eventually, it returns a message stating the pathname of the tar-ball archive (the filename is something like `ringclosingmm-None-py38_0.tar.bz2`) located under the conda-bld folder (search for the suggested argument to the `anaconda upload` command).
Copy the pathname to `.../envs/rco_package_builder/conda-bld`, we'll refer to it as `<path_to_conda-bld>`

```bash
# Create a virgin environment for this test
conda create -n rco_test
conda activate rco_test

# Require to use the conda-bld folder as a channel
conda config --add channels file://<path_to_conda-bld>

# Install the locally created Conda Package from the conda-bld channel
conda install --use-local ringclosingmm

# Veryfy the Installation
rc-optimized -h
```
