from pathlib import Path

def load_default_models():
    return {'full': 'yolov3.weights',
            'tiny': 'yolov3-tiny.weights'}


def load_labels(label_file):
    file = Path(label_file)
    with open(str(file), 'r') as f:
        classes = f.read().split('\n')[:-1]
    return classes

def anchors_for_yolov3(model_type='full'):
    if model_type == 'full':
        return [(10, 13), (16, 30), (33, 23),
                (30, 61), (62, 45), (59, 119),
                (116, 90), (156, 198), (373, 326)]
    else:
        return [(10, 14),  (23, 27),  (37, 58),
                (81, 82),  (135, 169),  (344, 319)]

def load_anchors(anchor_file):
	anchor_file = Path(anchor_file)
	if anchor_file.is_file():
		with open(anchor_file, 'rt') as f:
			line = f.readline()
			anchors_str = list(map(str.strip, line.split(',')))
			anchors_float = list(map(float, anchors_str))
			anchors_it = iter(anchors_float)
			anchors_tuple_list = list(zip(anchors_it, anchors_it))
		return anchors_tuple_list
	else:
		raise Exception('{} is not a valid path to anchor file!'.format(anchor_file))


def parse_args_from_txt(config_file):
    config_file = Path(config_file)
    if config_file.is_file():
        args = {}
        with open(config_file, 'rt') as f:
            for line in f:
                line = [t.strip() for t in line.split('=')]
                key = line[0]
                if key in ['width', 'height']:
                    value = int(line[1])
                elif key == 'labels':
                    value = load_labels(line[1])
                    args['num_classes'] = len(value)
                else:
                    value = line[1]
                args[key] = value
        return args                
    else:
        raise Exception('{} is not a valid path to config file!'.format(config_file))

if __name__ == "__main__":
    args = parse_args_from_txt('data/config/yolov3_full_416.cfg')
    print(args)
