import re
from tqdm import tqdm
import functools
import operator
import json

"""
uiview_original_subclasses = ['UIView', 'UILabel', 'UIPickerView',
                              'UIProgressView', 'UIActivityIndicatorView',
                              'UIImageView', 'UITabBar', 'UIToolBar',
                              'UINavigationBar', 'UIActionSheet', 'UIAlertView',
                              'UIScrollView', 'UISearchBar', 'UIWebView',
                              'UIButton', 'UIDatePicker', 'UIPageControl',
                              'UISegmentedControl', 'UITextField',
                              'UISlider', 'UISwitch']
"""
uiview_original_subclasses = ['UILabel', 'UIImageView', 'UITabBar', 'UIToolBar',
                              'UINavigationBar', 'UIAlertView', 'UISearchBar', 'UIWebView',
                              'UIButton', 'UIDatePicker', 'UIPageControl', 'UITextField',
                              'UISlider', 'UISwitch']


def generate_subclasses(texts):
    print('generate_subclasses')
    results = uiview_original_subclasses.copy()
    for line in tqdm(texts):
        for original_subclass in uiview_original_subclasses:
            regex = re.compile('\s(\w+)\s:\s' + original_subclass)
            results += regex.findall(line)
    print('subclasses: ' + str(len(results)))
    return results


def retrieve_entities_with_classes(texts, classes):
    print('retrieve_entities_with_classes')
    entities = {}
    cls_info = {}
    for cls in classes:
        entities[cls] = []
    for idx, line in tqdm(enumerate(texts)):
        for cls in classes:
            regex = re.compile(cls + '\s?\*\s?(\w+);')
            tmp = regex.findall(line)
            entities[cls] += [x for x in tmp if tmp.count(x) == 1]
            for entity in entities[cls]:
                cls_info[entity] = cls
    print('entities: ' + str(len(functools.reduce(operator.iconcat, list(entities.values()), []))))
    return entities, cls_info


def retrieve_code_blocks_with_entities(texts, entities):
    print('retrieve_code_blocks_with_entities')
    code_blocks = {}
    code_block_ids = {}
    for entity in entities:
        code_blocks[entity] = ""
    for idx, line in tqdm(enumerate(texts)):
        for entity in entities:
            regex = re.compile('\W{1}' + entity + '\W{1}')
            if len(regex.findall(line)) > 0:
                code_block_ids[entity] = idx
                code_blocks[entity] += line.strip() + " "
    return code_blocks, code_block_ids


def build_dataset():
    lines = open('../datasetProcessed/dataset_small.txt').readlines()
    classes = generate_subclasses(lines)
    entities, entity_cls_infos = retrieve_entities_with_classes(lines, classes)
    blocks, ids = retrieve_code_blocks_with_entities(lines,
                                                     functools.reduce(operator.iconcat, list(entities.values()), []))
    dataset = dict()
    dataset['data'] = {}
    dataset['data']['subclasses'] = classes
    dataset['data']['entities'] = []
    entities = functools.reduce(operator.iconcat, list(entities.values()), [])
    for entity in entities:
        try:
            tmp = dict()
            tmp['entity_name'] = entity
            tmp['entity_code_block'] = blocks[entity]
            tmp['entity_line_num'] = ids[entity]
            tmp['entity_class'] = entity_cls_infos[entity]
            dataset['data']['entities'].append(tmp)
        except:
            continue
    return dataset


if __name__ == '__main__':
    dataset = build_dataset()
    f = open('data.json', 'w')
    json.dump(dataset, f)
