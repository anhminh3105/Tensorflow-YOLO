from pathlib import Path

def load_class_names(path):
    file = Path(path)
    with open(file, 'r') as f:
        classes = f.read().split('\n')[:-1]
    
    return classes