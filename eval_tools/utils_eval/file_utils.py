import os, json

# Write data to a JSON file
def write_json(path, res):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, 'w', encoding='utf-8') as f:
        data = json.dumps(res, ensure_ascii=False, indent=2)
        f.write(data)

# Load a JSON file from the given path
def load_json(path):
    try:
        with open(path, 'r') as f:
            res = json.load(f)
    except FileNotFoundError:
        res = None
    return res