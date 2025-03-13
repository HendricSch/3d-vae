# Description: Run sanity check for the project


# Activate the virtual environment
echo "Activating the virtual environment..."
./venv/Scripts/activate

# Run the sanity check
echo "Running the sanity check..."
python -m sanity_check > sanity_check.log

echo "Sanity check completed. Check sanity_check.log for details."
