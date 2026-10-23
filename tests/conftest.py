import os
import sys

# The qsartools package lives in scripts/; make it importable as `qsartools.<module>`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
