# cleanup_dataset.py
# -------------------------------------------------------
# Removes:
# 1. Corrupted images
# 2. Duplicate images
# from datasets/skin_conditions/
# -------------------------------------------------------

from pathlib import Path
from PIL import Image
import hashlib
import os

# -------------------------------------------------------
# DATASET PATH
# -------------------------------------------------------

DATASET_DIR = Path("datasets/skin_conditions")

# -------------------------------------------------------
# SUPPORTED IMAGE EXTENSIONS
# -------------------------------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# -------------------------------------------------------
# STORAGE
# -------------------------------------------------------

hashes = {}
corrupted_files = []
duplicate_files = []

# -------------------------------------------------------
# HASH FUNCTION
# -------------------------------------------------------

def file_hash(path: Path):
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

# -------------------------------------------------------
# IMAGE VALIDATION
# -------------------------------------------------------

def is_valid_image(path: Path):
    try:
        with Image.open(path) as img:
            img.verify()

        # reopen fully
        with Image.open(path) as img:
            img.convert("RGB")

        return True

    except Exception:
        return False

# -------------------------------------------------------
# MAIN CLEANUP
# -------------------------------------------------------

print("\nScanning dataset...\n")

all_images = [
    p for p in DATASET_DIR.rglob("*")
    if p.suffix.lower() in VALID_EXTENSIONS
]

print(f"Found {len(all_images)} image files.\n")

for path in all_images:

    # ---------------------------------------------------
    # CHECK CORRUPTION
    # ---------------------------------------------------

    if not is_valid_image(path):
        corrupted_files.append(path)
        continue

    # ---------------------------------------------------
    # CHECK DUPLICATES
    # ---------------------------------------------------

    try:
        h = file_hash(path)

        if h in hashes:
            duplicate_files.append(path)
        else:
            hashes[h] = path

    except Exception:
        corrupted_files.append(path)

# -------------------------------------------------------
# DELETE CORRUPTED
# -------------------------------------------------------

print(f"\nCorrupted images found: {len(corrupted_files)}")

for path in corrupted_files:
    try:
        os.remove(path)
        print(f"Removed corrupted: {path}")
    except Exception as e:
        print(f"Failed removing {path}: {e}")

# -------------------------------------------------------
# DELETE DUPLICATES
# -------------------------------------------------------

print(f"\nDuplicate images found: {len(duplicate_files)}")

for path in duplicate_files:
    try:
        os.remove(path)
        print(f"Removed duplicate: {path}")
    except Exception as e:
        print(f"Failed removing {path}: {e}")

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------

print("\n-----------------------------------")
print("DATASET CLEANUP COMPLETE")
print("-----------------------------------")
print(f"Removed corrupted : {len(corrupted_files)}")
print(f"Removed duplicates: {len(duplicate_files)}")
print("-----------------------------------")