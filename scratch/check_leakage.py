import os
import sys
import numpy as np
import pandas as pd
import json

notebook_path = '/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent/notebooks/02_technical_ml_experiment.ipynb'

# Let's inspect the code in the notebook
with open(notebook_path, 'r') as f:
    nb = json.load(f)

print("Checking notebook structure...")
