import os
from PIL import Image
import imagehash
from tqdm import tqdm

def clean_duplicates(folder_path):
    print(f"\nCleaning: {folder_path}")
    
    hashes = {}
    removed = 0
    total = 0

    for root, _, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
                path = os.path.join(root, file)

                try:
                    img = Image.open(path).convert('RGB')
                    img_hash = imagehash.phash(img)

                    if img_hash in hashes:
                        os.remove(path)
                        removed += 1
                    else:
                        hashes[img_hash] = path

                except Exception as e:
                    print(f"Error with {path}: {e}")

    print(f"Total images: {total}")
    print(f"Duplicates removed: {removed}")
    print(f"Remaining: {total - removed}")


if __name__ == "__main__":
    clean_duplicates("datasets/skin_issues")
    clean_duplicates("datasets/skin_type")

    print("\n Cleaning completed successfully!")