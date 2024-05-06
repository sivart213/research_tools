# Research_Tools
Package for generic functions and equations compiled from various sources during my PhD career.

## Description
This is a non-production package which is used by my other packages.  

## Installation

Option 1: download the repository to your code folder and pip install via:

    pip install -e "path\to\your\python\project"

Option 2: pip install directly from github

    pip install "git+https://github.com/sivart213/research_tools#egg=research_tools"

If using conda, be sure install in a separate environment to protect your installation.

### Alternate
Code should also be available for use if "research_tools" folder is placed into your own directory


## Contents
- `utils`: Contains project management scripts unrelated to the library
    - `init_util.py`: Create str for `__init__.py` files or push updates to all packages
    - `gen_package_list.py`: create an env.yml
- `equations`: Contains scripts w/ basic equations which can be called an manipulated
- `functions`: Contains useful operational functions to interface with data or os