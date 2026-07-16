import urllib.request, tarfile, os, sys

url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
out = "/tmp/dev-clean.tar.gz"
print("Downloading", url)
urllib.request.urlretrieve(url, out)
print("Downloaded to", out)
os.makedirs("/tmp/LibriSpeech", exist_ok=True)
print("Extracting...")
with tarfile.open(out, "r:gz") as t:
    t.extractall("/tmp")
print("Extracted to /tmp/LibriSpeech")
print("Done")
