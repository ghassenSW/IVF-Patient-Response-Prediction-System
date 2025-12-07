#!/usr/bin/env python3
"""
Quick script to check if model files exist
Run this during Render build to verify files are present
"""
import os
from pathlib import Path

print("=" * 60)
print("FILE EXISTENCE CHECK")
print("=" * 60)

# Get current working directory
cwd = Path.cwd()
print(f"\nCurrent working directory: {cwd}")

# List contents of current directory
print(f"\nContents of {cwd}:")
for item in sorted(cwd.iterdir()):
    print(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")

# Check for src directory
src_dir = cwd / "src"
if src_dir.exists():
    print(f"\n✓ src directory exists: {src_dir}")
    print(f"Contents of src:")
    for item in sorted(src_dir.iterdir()):
        print(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
    
    # Check for model directory
    model_dir = src_dir / "model"
    if model_dir.exists():
        print(f"\n✓ model directory exists: {model_dir}")
        print(f"Contents of model:")
        for item in sorted(model_dir.iterdir()):
            print(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        
        # Check for saved_models
        saved_models_dir = model_dir / "saved_models"
        if saved_models_dir.exists():
            print(f"\n✓ saved_models directory exists: {saved_models_dir}")
            print(f"Contents of saved_models:")
            for item in sorted(saved_models_dir.iterdir()):
                size = item.stat().st_size if item.is_file() else 0
                print(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name} ({size:,} bytes)")
        else:
            print(f"\n✗ saved_models directory NOT FOUND: {saved_models_dir}")
    else:
        print(f"\n✗ model directory NOT FOUND: {model_dir}")
else:
    print(f"\n✗ src directory NOT FOUND: {src_dir}")

# Check for data directory
data_dir = cwd / "data"
if data_dir.exists():
    print(f"\n✓ data directory exists: {data_dir}")
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        print(f"✓ processed directory exists: {processed_dir}")
        print(f"Contents of data/processed:")
        for item in sorted(processed_dir.iterdir()):
            size = item.stat().st_size if item.is_file() else 0
            print(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name} ({size:,} bytes)")
    else:
        print(f"✗ processed directory NOT FOUND: {processed_dir}")
else:
    print(f"\n✗ data directory NOT FOUND: {data_dir}")

print("\n" + "=" * 60)
