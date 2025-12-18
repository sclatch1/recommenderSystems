#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-file-or-name>"
  exit 1
fi

input="$1"
base="$(basename "$input")"

cd output

zip "zip_folders/${base}_recommendations.csv.zip" "${base}.csv"
echo "Created zip_folders/${base}_recommendations.csv.zip"