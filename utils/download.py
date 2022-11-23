import os
import shutil
import urllib
import pathlib
import tarfile
import zipfile


def download_data(out_path, url, extract=True, force=False):
    pathlib.Path(out_path).mkdir(exist_ok=True)
    out_filename = os.path.join(out_path, os.path.basename(url))

    if os.path.isfile(out_filename) and not force:
        print(f"File {out_filename} exists, skipping download.")
    else:
        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as response:
            with open(out_filename, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

        print(f"Saved to {out_filename}.")

    extracted_dir = None
    if extract and out_filename.endswith(".zip"):
        print(f"Extracting {out_filename}...")
        with zipfile.ZipFile(out_filename, "r") as zipf:
            names = zipf.namelist()
            zipf.extractall(out_path)
            zipinfos = zipf.infolist()
            first_dir = next(filter(lambda zi: zi.is_dir(), zipinfos)).filename
            extracted_dir = os.path.join(out_path, os.path.dirname(first_dir))
            print(f"Extracted {len(names)} to {extracted_dir}")
            retval = extracted_dir

    if extract and out_filename.endswith((".tar.gz", ".tgz")):
        print(f"Extracting {out_filename}...")
        with tarfile.open(out_filename, "r") as tarf:
            members = tarf.getmembers()
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tarf, out_path)
            first_dir = next(filter(lambda ti: ti.isdir(), members)).name
            extracted_dir = os.path.join(out_path, os.path.dirname(first_dir))
            print(f"Extracted {len(members)} to {extracted_dir}")
            retval = extracted_dir

    return out_filename, extracted_dir
