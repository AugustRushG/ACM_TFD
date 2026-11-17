import json
import gzip

def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)



def save_to_json(data, file_path):
    """Saves a dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)