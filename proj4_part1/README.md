# CS 6476 Project 4: [Depth Estimation using Stereo]((https://www.cc.gatech.edu/~hays/compvision/proj4/proj4_part1.pdf))

## Setup
1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
2. Download and extract the part 1 starter code.
3. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj4_env_<OS>.yml`. If you're running into issues building the environment, try running `conda update --all` first.
4. This should create an environment named 'proj4'. Activate it using the Windows command, `activate cs6476_proj4` or the MacOS / Linux command, `source activate cs6476_proj4`, `conda activate cs6476_proj4`
5. Install the project package, by running `pip install -e .` inside the repo folder.
6. Run the notebook using `jupyter notebook ./proj4_code/part1_simple_stereo.ipynb`
7. After implementing all functions, ensure that all sanity checks are passing by running `pytest proj4_unit_tests` inside the repo folder.
8. Complete part 2 (template and instructions to be released separately).