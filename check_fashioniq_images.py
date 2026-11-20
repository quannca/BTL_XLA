
from pathlib import Path
from PIL import Image

folder = Path("data/fashion_iq/images_recovered")

files = list(folder.glob("*.jpg"))
bad = []

for file in files:
    try:
        with Image.open(file) as img:
            img.verify()
    except Exception as e:
        bad.append((file.name, str(e)))

print("Total JPG:", len(files))
print("Valid images:", len(files) - len(bad))
print("Unreadable images:", len(bad))

if bad:
    print("\nFirst 20 corrupted files:")
    for name, error in bad[:20]:
        print(name, error)
