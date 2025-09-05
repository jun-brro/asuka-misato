#!/usr/bin/env python3
import os
import requests
import huggingface_hub
from tqdm import tqdm
import argparse

def download_file(url, destination):
    """Download a file from a URL to a destination with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists and has the same size
    if os.path.exists(destination):
        file_size = os.path.getsize(destination)
        if file_size == total_size and total_size > 0:
            print(f"File already exists and has correct size: {destination}")
            return
    
    # Download with progress bar
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_from_huggingface(repo_id, output_dir="ckpt"):
    """Download all files from a Hugging Face repository."""
    print(f"Downloading models from {repo_id} to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files in the repository
    files = huggingface_hub.list_repo_files(repo_id)
    
    # Filter out directories and non-model files if needed
    model_files = [f for f in files if not f.endswith('/') and not f.startswith('.')]
    
    # Download each file
    for file_path in model_files:
        file_url = huggingface_hub.hf_hub_url(repo_id=repo_id, filename=file_path)
        local_path = os.path.join(output_dir, os.path.basename(file_path))
        
        print(f"Downloading {file_path}...")
        download_file(file_url, local_path)
    
    print("Download complete!")

def main():
    parser = argparse.ArgumentParser(description="Download ASUKA-FLUX models from Hugging Face")
    parser.add_argument("--output_dir", type=str, default="ckpt", 
                        help="Directory to save the downloaded models")
    args = parser.parse_args()
    
    # Download models from the Hugging Face repository
    repo_id = "yikaiwang/ASUKA-FLUX.1-Fill"
    download_from_huggingface(repo_id, args.output_dir)
    
    # Check if the required model file exists
    required_model = os.path.join(args.output_dir, "asuka_decoder.ckpt")
    if os.path.exists(required_model):
        print(f"Successfully downloaded the required model: {required_model}")
    else:
        print(f"Warning: Required model file {required_model} was not found.")
        print("Please check the Hugging Face repository for the correct file names.")

if __name__ == "__main__":
    main()
