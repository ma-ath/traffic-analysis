"""
/--------------------------------/
Created on June 1, 2021
@authors:
    Matheus S. Lima
@company: Federal University of Rio de Janeiro - Polytechnic School - Analog and Digital Signal Processing Laboratory (PADS)
/--------------------------------/
"""

class TrafficSoundAnalysis:
    """
    Super class that contains the code for all projects related to city traffic analysis.
    """

    def __init__(self):
        """
        Class constructor
        """
        pass

    pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PADS library for traffic analysis')
    args = parser.parse_args()