import lmdb
import os

# Path to the LMDB file
lmdb_path = '/root/latent-diffusion/ldm/data/church_outdoor_train_lmdb'
# Directory where you want to save the extracted images
output_dir = 'data/lsun/churches/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

env = lmdb.open(lmdb_path, readonly=True, max_dbs=0, lock=False, readahead=False, meminit=False)

entry_count = 0

with env.begin(write=False) as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        entry_count += 1
        # The key is typically the filename. Adjust the key or filename as necessary.
        filename = key.decode('utf-8') + '.webp'
        output_path = os.path.join(output_dir, filename)
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(value)

env.close()

print(f"Total entries processed: {entry_count}")
