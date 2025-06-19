import os
import pickle

def ensure_dir(path):
    """Ensure the given directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_pickle(obj, path):
    """Save an object to disk using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """Load an object from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_lines(file_path):
    """Read a text file line by line."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def write_lines(file_path, lines):
    """Write a list of strings to a file."""
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")