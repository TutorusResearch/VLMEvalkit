import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate from VLMEvalKit/prepare_kor_dataset to VLMEvalKit/playground
output_dir = os.path.join(current_dir, "..", "playground")
output_dir = os.path.abspath(output_dir)  # Normalize to absolute path
os.makedirs(output_dir, exist_ok=True)

# List of datasets to download
datasets = [
    "Tutoruslabs/ChartQA_KOR",
    "Tutoruslabs/ELEMENTARY_MATH",
    "Tutoruslabs/KMMVisMath"
]

print("=" * 60)
print("Downloading Korean Math Datasets")
print("=" * 60)

total_downloaded = 0

# Process each dataset
for repo_id in datasets:
    print(f"\n### Processing: {repo_id}")

    # List all files in the repository
    all_files = list_repo_files(repo_id, repo_type="dataset")

    # Filter only .tsv files in for_eval/ folder
    tsv_files = [f for f in all_files if f.startswith("for_eval/") and f.endswith(".tsv")]

    if not tsv_files:
        print(f"  No TSV files found in for_eval/ folder")
        continue

    print(f"  Found {len(tsv_files)} TSV file(s):")
    for file in tsv_files:
        print(f"    - {file}")

    # Download each file
    for file_path in tsv_files:
        # Extract filename only (remove for_eval/ prefix)
        filename = os.path.basename(file_path)
        final_path = os.path.join(output_dir, filename)

        print(f"\n  Downloading {file_path}...")

        # Download file directly (it will be downloaded to cache)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset"
        )

        # Copy to playground directory
        shutil.copy2(downloaded_path, final_path)
        print(f"  Saved to: {final_path}")

        total_downloaded += 1

print("\n" + "=" * 60)
print(f"Download completed! Total {total_downloaded} file(s) downloaded.")
print(f"Files saved to: {output_dir}")
print("=" * 60)
