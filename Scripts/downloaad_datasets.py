#!/usr/bin/env python

import os
import shutil
import subprocess
import sys

# Check if gdown is installed
if not shutil.which("gdown"):
    print("gdown is not installed. Installing gdown...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"])

# Set the file ID to download
file_id = "1KKpc5nfZLs4lVCWFVtdrJazR7BidioRX"

# Set the destination folder
destination_folder = "data"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)

# Download the file using gdown
print("Downloading the file...")
subprocess.run(["gdown", f"https://drive.google.com/uc?id={file_id}", "-O", f"{destination_folder}/archive (4).zip"])

# Unzip the file to the dataset folder
print("Unzipping the file...")
subprocess.run(["unzip", f"{destination_folder}/archive (4).zip", "-d", destination_folder])

# Remove the downloaded zip file
print("Removing the zip file...")
os.remove(f"{destination_folder}/archive (4).zip")

print("File downloaded, unzipped, and zip file removed successfully.")