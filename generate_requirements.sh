#!/bin/bash

# Generate requirements.txt using pipreqs and pip freeze
# This script runs the commands in the virtual environment and then processes the results

set -e  # Exit on any error

echo "🔧 Generating requirements.txt with correct versions..."

# Step 1: Run pipreqs
echo ""
echo "1️⃣  Running pipreqs..."
if ! pipreqs . --force --savepath reqs_from_pipreqs.txt 2>&1; then
    echo "❌ Failed to run pipreqs. Make sure it's installed: pip install pipreqs"
    exit 1
fi
echo "✓ Successfully generated reqs_from_pipreqs.txt"

# Step 2: Run pip freeze
echo ""
echo "2️⃣  Running pip freeze..."
if ! pip freeze > reqs_from_pipfreeze.txt 2>&1; then
    echo "❌ Failed to run pip freeze"
    exit 1
fi
echo "✓ Successfully generated reqs_from_pipfreeze.txt"

# Step 3: Run the Python script to process the files
echo ""
echo "3️⃣  Processing files with Python script..."
if ! python3 generate_requirements.py; then
    echo "❌ Failed to run generate_requirements.py"
    exit 1
fi

echo ""
echo "✅ Done! Check requirements.txt for the final result." 