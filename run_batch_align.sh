#!/bin/bash

# This script runs the unialign_v3.py alignment for all protein pairs
# specified in id_pair.txt and saves the output to individual log files.

# --- Configuration ---
# The parameters for the alignment script
ALPHA=0.9
REG_M=0.1
EPSILON=0.001

# --- Script Start ---

# Create a directory for the log files if it doesn't exist
LOG_DIR="output/alignment_logs"
mkdir -p "$LOG_DIR"

echo "Starting batch alignment process..."
echo "Log files will be saved in $LOG_DIR"

# Check if id_pair.txt exists
if [ ! -f "id_pair.txt" ]; then
    echo "Error: id_pair.txt not found!"
    exit 1
fi

# Read id_pair.txt line by line
while read -r p1 p2; do
  # Skip empty or invalid lines
  if [ -z "$p1" ] || [ -z "$p2" ]; then
    continue
  fi

  # Construct file paths
  file1="RPIC_all/${p1}.pdb"
  file2="RPIC_all/${p2}.pdb"
  log_file="${LOG_DIR}/${p1}_vs_${p2}.log"

  echo "--------------------------------------------------"
  echo "Queueing alignment for ${p1} and ${p2}..."
  echo "Log will be at: ${log_file}"

  # Check if input files exist before running
  if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
      echo "Warning: Skipping pair, one or both PDB files not found: $file1, $file2" | tee -a "$log_file"
      continue
  fi

  # Run the command and redirect all output (stdout and stderr) to the log file
  python unialign_v6.py "$file1" "$file2" --alpha $ALPHA --reg_m $REG_M --epsilon $EPSILON > "$log_file" 2>&1

  # Optional: Check the exit code of the last command
  if [ $? -eq 0 ]; then
    echo "Alignment for ${p1} and ${p2} completed successfully."
  else
    echo "Alignment for ${p1} and ${p2} failed. Check log file for details."
  fi

done < id_pair.txt

echo "--------------------------------------------------"
echo "Batch alignment process finished."
