#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-file-or-name>"
  exit 1
fi

input="$1"

if [ -f "$input" ]; then
  file="$input"
elif [ -f "output/${input}.csv" ]; then
  file="output/${input}.csv"
elif [ -f "output/${input}" ]; then
  file="output/${input}"
else
  echo "File not found: $input (also checked output/${input}.csv)"
  exit 1
fi

base="$(basename "$file" .csv)"
mkdir -p zip_folders

if zip -j "zip_folders/${base}_recommendations.csv.zip" "$file"; then
  echo "Created zip_folders/${base}_recommendations.csv.zip"
else
  echo "Failed to create zip_folders/${base}_recommendations.csv.zip"
  exit 1
fi
