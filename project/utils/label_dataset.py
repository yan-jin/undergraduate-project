import json
from collections import Counter


def label_dataset(path, output_path):
    data = json.load(open(path, 'r'))
    entities = data['data']['entities']
    data['data']['id2label'] = {
        0: 'UIAccessibilityTraitNone',
        1: 'UIAccessibilityTraitButton',
        2: 'UIAccessibilityTraitLink',
        3: 'UIAccessibilityTraitHeader',
        4: 'UIAccessibilityTraitSearchField',
        5: 'UIAccessibilityTraitImage',
        6: 'UIAccessibilityTraitSelected',
        7: 'UIAccessibilityTraitPlaysSound',
        8: 'UIAccessibilityTraitKeyboardKey',
        9: 'UIAccessibilityTraitStaticText',
        10: 'UIAccessibilityTraitSummaryElement',
        11: 'UIAccessibilityTraitNotEnabled',
        12: 'UIAccessibilityTraitUpdatesFrequently',
        13: 'UIAccessibilityTraitStartsMediaSession',
        14: 'UIAccessibilityTraitAdjustable',
        15: 'UIAccessibilityTraitAllowsDirectInteraction',
        16: 'UIAccessibilityTraitCausesPageTurn'
    }
    data['data']['label2id'] = dict()
    for key, value in data['data']['id2label'].items():
        data['data']['label2id'][value] = key

    def save():
        print('File saved...')
        data['data']['entities'] = entities
        json.dump(data, open(output_path, 'w'))

    for idx, entity in enumerate(entities):
        try:
            if 'label_type' in entity.keys():
                pass# continue
            print('\n')
            print(str(idx + 1) + ' in ' + str(len(entities)))
            print('[entity_name]: ' + str(entity['entity_name']) + '\n')
            print('[entity_class]: ' + str(entity['entity_class']) + '\n')
            print('[entity_code_block]:\n\n' + str(entity['entity_code_block']) + '\n')
            label_type = input('label_type:')
            if label_type.isnumeric():
                entity['label_type'] = int(label_type)
            else:
                save()
                return data
        except KeyboardInterrupt:
            save()
            return data
    save()

    return data


def dataset_summary(path):
    data = json.load(open(path, 'r'))
    entities = data['data']['entities']
    id2label = data['data']['id2label']
    lbl_cnt = Counter()
    for entity in entities:
        lbl_cnt[id2label[str(entity['label_type'])]] += 1

    for item in lbl_cnt.items():
        print(item)


if __name__ == '__main__':
    label_dataset('/Users/jinyan/Documents/YanJin/finalproject/project/sample_data.json', './sample_data.json')
    dataset_summary('/Users/jinyan/Documents/YanJin/finalproject/project/sample_data.json')
