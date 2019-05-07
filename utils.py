from pathlib import Path

def load_file_names():
    return {'full': 'yolov3.weights',
            'tiny': 'yolov3-tiny.weights'}

def load_class_names(path):
    file = Path(path)
    with open(str(file), 'r') as f:
        classes = f.read().split('\n')[:-1]
    
    return classes

def anchors_for_yolov3(model_type='full'):
    if model_type is 'full':
        return [(10, 13), (16, 30), (33, 23),
                (30, 61), (62, 45), (59, 119),
                (116, 90), (156, 198), (373, 326)]
    else:
        return [(10, 14),  (23, 27),  (37, 58),
                (81, 82),  (135, 169),  (344, 319)]
