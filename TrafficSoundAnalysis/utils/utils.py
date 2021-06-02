"""
/--------------------------------/
Created on June 1, 2021
@authors:
    Matheus S. Lima
@company: Federal University of Rio de Janeiro - Polytechnic School - Analog and Digital Signal Processing Laboratory (PADS)
/--------------------------------/
"""

from shutil import which

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_info(s):
    print(f"{bcolors.OKGREEN}[INFO]: {bcolors.ENDC}"+s)

def print_error(s):
    print(f"{bcolors.FAIL}[ERROR]: {bcolors.ENDC}"+s)

def print_warning(s):
    print(f"{bcolors.WARNING}[WARNING]: {bcolors.ENDC}"+s)

def is_tool(name):
    """
        Check whether `name` is on PATH and marked as executable.
    """
    return which(name) is not None