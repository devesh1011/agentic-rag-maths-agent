#!/usr/bin/env bash
# This script ensures that the build process will exit immediately if any command fails.
set -o errexit

# 1. Install System Dependencies
# This section updates the package list and installs the C libraries (libxml2, libxslt)
# that are required to compile the lxml Python package from source.
echo "Installing system dependencies for lxml..."
apt-get update
apt-get install -y libxml2-dev libxslt1-dev

# 2. Install Python Dependencies
# After the system is prepared, this command installs all the Python packages
# listed in your requirements.txt file.
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Build finished successfully!"
