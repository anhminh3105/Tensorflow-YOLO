from pathlib import Path

def load_class_names(path):
    file = Path(path)
    with open(str(file), 'r') as f:
        classes = f.read().split('\n')[:-1]
    
    return classes

def anchors_for_yolov3():
    return [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]

def arg_find(list, value):
    'find the index of value in list.'
    return list.index(value) if value in list else None