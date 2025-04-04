#!/bin/bash

DIR="outputs-task/"

# Loop through all files in the directory
for file in "$DIR/*; do
    # Skip if it's a directory
    if [ -f "$file" ]; then
        # Get just the filename without path
        fileid=$(basename "$file")
        
        echo "Processing: $fileid"
        
        # Launch Python script with the filename as experiment_name
        python resume_experiment_remote.py --experiment_name "$fileid" --save_path "$DIR" 
        
        echo ""
    fi
done

echo "All experiments have been processed."