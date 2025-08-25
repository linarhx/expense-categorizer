#!/usr/bin/env python3
"""
Launcher script for the Expense Categorizer Streamlit app.
This script checks if the model exists and provides options to train or run the app.
"""

import os
import subprocess
import sys

def main():
    print("💰 Expense Categorizer")
    
    if not os.path.exists('models/model.joblib'):
        print("Training model first...")
        subprocess.run([sys.executable, 'src/train.py'])
    
    print("Starting Streamlit app...")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app/main.py'])

if __name__ == "__main__":
    main()
