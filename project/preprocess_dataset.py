import os
import shutil
from tqdm import tqdm


def is_target_file(filename):
    return filename[-2:] == ".h" or filename[-2:] == ".m"


def process_path(root_path, cur_path):
    cur_subpaths = os.listdir(cur_path)
    for sub_path in cur_subpaths:
        sub_path = os.path.join(cur_path, sub_path)
        if os.path.isfile(sub_path) and is_target_file(sub_path):
            shutil.copy(sub_path, root_path)
            print("copy[" + sub_path + "] to [" + root_path + "]")
        elif os.path.isdir(sub_path):
            process_path(root_path, sub_path)


def combine_files(root, output_path):
    paths = os.listdir(root)
    data = []
    for path in tqdm(paths):
        path = os.path.join(root, path)
        if os.path.isfile(path) and is_target_file(path):
            try:
                f = open(path, 'r')
                lines = f.readlines()
                data += lines
                f.close()
                os.remove(path)
            except:
                continue
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    f = open(os.path.join(output_path, 'combined_data.txt'), 'w')
    f.writelines(data)
