#!/usr/bin/env python3
"""
Download MISATO Dataset from Hugging Face
Author: Assistant
Based on: https://huggingface.co/datasets/yikaiwang/MISATO
"""

import os
import argparse
from huggingface_hub import snapshot_download
import zipfile
from tqdm import tqdm

def download_misato_dataset(output_dir="./data", cache_dir=None):
    """
    Download MISATO dataset from Hugging Face.
    
    Note: You need to accept the terms and conditions on the Hugging Face page first:
    https://huggingface.co/datasets/yikaiwang/MISATO
    
    Args:
        output_dir: Directory to extract the dataset
        cache_dir: Directory to cache the downloaded files
    """
    
    print("=" * 60)
    print("MISATO Dataset Download")
    print("=" * 60)
    print(f"Repository: https://huggingface.co/datasets/yikaiwang/MISATO")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if user has agreed to terms
    print("IMPORTANT: Before downloading, you must:")
    print("1. Go to https://huggingface.co/datasets/yikaiwang/MISATO")
    print("2. Log in to your Hugging Face account")
    print("3. Accept the terms and conditions to access the dataset")
    print()
    
    response = input("Have you completed the above steps? (y/N): ")
    if response.lower() != 'y':
        print("Please complete the required steps first.")
        return False
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        print("Downloading MISATO dataset...")
        repo_id = "yikaiwang/MISATO"
        
        # Download to cache directory first
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"Dataset downloaded to: {local_dir}")
        
        # Look for zip files and extract them
        zip_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith('.zip'):
                    zip_files.append(os.path.join(root, file))
        
        if zip_files:
            print(f"Found {len(zip_files)} zip file(s) to extract...")
            for zip_file in zip_files:
                print(f"Extracting {os.path.basename(zip_file)}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extract with progress bar
                    members = zip_ref.infolist()
                    for member in tqdm(members, desc="Extracting"):
                        zip_ref.extract(member, output_dir)
        else:
            # If no zip files, copy the files directly
            import shutil
            print("Copying dataset files...")
            if os.path.exists(os.path.join(local_dir, "512")):
                shutil.copytree(os.path.join(local_dir, "512"), os.path.join(output_dir, "512"), dirs_exist_ok=True)
            if os.path.exists(os.path.join(local_dir, "1024")):
                shutil.copytree(os.path.join(local_dir, "1024"), os.path.join(output_dir, "1024"), dirs_exist_ok=True)
        
        # Verify the dataset structure
        verify_dataset_structure(output_dir)
        
        print("\n" + "=" * 60)
        print("Dataset download completed successfully!")
        print("=" * 60)
        print(f"Dataset location: {output_dir}")
        print("\nDataset structure:")
        print("- 512/ : 512x512 resolution images (2000 images)")
        print("  - image/: Original images (0000.png - 1999.png)")
        print("  - mask/: Mask images (0000.png - 1999.png)")
        print("- 1024/ : 1024x1024 resolution images (1500 images)")
        print("  - image/: Original images")
        print("  - mask/: Mask images")
        print("\nImage categories:")
        print("- 0000-0499: Outdoor landscapes")
        print("- 0500-0999: Indoor scenes")
        print("- 1000-1499: Buildings")
        print("- 1500-1999: Backgrounds")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in to Hugging Face:")
        print("   huggingface-cli login")
        print("2. Make sure you've accepted the dataset terms at:")
        print("   https://huggingface.co/datasets/yikaiwang/MISATO")
        return False

def verify_dataset_structure(data_dir):
    """Verify that the dataset has the expected structure."""
    
    print("\nVerifying dataset structure...")
    
    expected_dirs = [
        "512/image",
        "512/mask", 
        "1024/image",
        "1024/mask"
    ]
    
    for dir_path in expected_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if os.path.exists(full_path):
            file_count = len([f for f in os.listdir(full_path) if f.endswith('.png')])
            print(f"✓ {dir_path}: {file_count} files")
        else:
            print(f"✗ {dir_path}: Not found")

def main():
    parser = argparse.ArgumentParser(description="Download MISATO Dataset from Hugging Face")
    parser.add_argument("--output_dir", type=str, default="./data", 
                        help="Directory to extract the dataset (default: ./data)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache downloaded files")
    
    args = parser.parse_args()
    
    success = download_misato_dataset(args.output_dir, args.cache_dir)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
