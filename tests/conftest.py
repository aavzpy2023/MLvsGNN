import os
import sys

# The tools live in scripts/tools; make them importable as `tools.<module>`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
