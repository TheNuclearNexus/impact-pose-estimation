import os
import requests
import sys


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)


def main():
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    checkpoint_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    download_file(checkpoint_url, checkpoint_path)


if __name__ == "__main__":
    main()
