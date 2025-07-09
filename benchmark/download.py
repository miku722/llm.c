import requests
import os
import tarfile

def download_file(url, dest_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def extract_tar_gz(filepath, extract_to):
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_to)

urls = [
    ("https://zenodo.org/records/2630551/files/lambada-dataset.tar.gz?download=1", "lambada-dataset.tar.gz"),
    ("http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz", "CBTest.tgz"),
]

output_dir = "benchmark/datasets"
os.makedirs(output_dir, exist_ok=True)

for url, filename in urls:
    dest_path = os.path.join(output_dir, filename)
    print(f"⬇ Downloading {filename}...")
    download_file(url, dest_path)
    print(f"✅ Downloaded to {dest_path}")

    if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        print(f"📦 Extracting {filename}...")
        extract_tar_gz(dest_path, output_dir)
        print(f"✅ Extracted to {output_dir}")
