import time
import random
import os 

def downloads(package_name, size_in_mb, total_time_sec):
    print(f"Downloading {package_name} ({size_in_mb} MB)...")
    total_chunks = 50  # Number of chunks for the progress bar
    chunk_time = total_time_sec / total_chunks  # Time per chunk
    downloaded = 0
    chunk_size = size_in_mb / total_chunks  # Simulate chunk size in MB

    for i in range(total_chunks):
        time.sleep(chunk_time)  # Simulate the time for each chunk
        downloaded += chunk_size
        progress = int((i + 1) / total_chunks * 100)
        bar = "#" * (i + 1) + "-" * (total_chunks - i - 1)
        print(f"\r[{bar}] {progress}% ({downloaded:.1f}/{size_in_mb:.1f} MB)", end="", flush=True)

    print(f"\n{package_name} downloaded successfully!")

def time_to_seconds(minutes, seconds):
    """Convert minutes and seconds to total seconds."""
    """The conversion takes place in real time .
    """
    return minutes * 60 + seconds


def main():
    
    paks = [
        ("spacy_nlp_model", 1219.74, (2, 33)),  # 2 minutes 33 seconds
        ("resnet_dataset_large", 836.31, (1, 45)),  # 1 minute 45 seconds
        ("deep_neura_2160", 82.33, (3, 20)),  # 3 minutes 20 seconds
        ("adam_optimizer", 610.24, (1, 0)),  # 1 minute
        ("quantum_simulation_library", 951, (2, 15)),  # 2 minutes 15 seconds
    ]
    with open("package.py", "r") as file:
        print(file.read())
    

    for package_name, size, (minutes, seconds) in paks:
        total_time_sec = time_to_seconds(minutes, seconds)
        downloads(package_name, size, total_time_sec)
        print()

if __name__ == "__main__":
    main()
