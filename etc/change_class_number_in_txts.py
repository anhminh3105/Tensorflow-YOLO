import os
from pathlib import Path
from argparse import ArgumentParser

def arg_builder():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-d', '--dir_path', 
                            help='path to the directory where you store all training .txt files.',
                            required='True',
                            type=str)
    arg_parser.add_argument('-c', '--class_num',
                            help='the class number you want to change',
                            required=True,
                            type=int)
    return arg_parser.parse_args()

def change_class_number_in_txts(args):
    dir_path = Path(args.dir_path)
    class_num = args.class_num

    txts = [txt for txt in os.listdir(dir_path) if txt.endswith('.txt')]

    for txt in txts:
        content = []
        txt_path = dir_path.joinpath(txt)
        with open(txt_path) as t:
            for line in t:
                line_list = line.split()
                if line_list[0] != class_num:
                    line_list[0] = str(class_num)
                new_line = ' '.join(line_list)
                content.append(new_line)
        with open(txt_path, 'w') as t:
            t.writelines('{}\n'.format(line) for line in content)

if __name__ == "__main__":
    change_class_number_in_txts(arg_builder())
    print('Done, bye!')    

