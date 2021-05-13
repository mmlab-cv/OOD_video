from __future__ import print_function, division
import os
import sys
import json
import pandas as pd

def convert_csv_to_dict(csv_dir_path, split_index,is_test):
    database = {}
    for filename in os.listdir(csv_dir_path):
        
        data = pd.read_csv(os.path.join(csv_dir_path, filename),
                           delimiter=' ', header=None)
        keys = []
        subsets = []
        for i in range(data.shape[0]):
            row = data.loc[i, :]
            # print('row ',row[0])
            # if row[1] == 0:
            #     continue
            # elif row[1] == 1:
            #     subset = 'training'
            # elif row[1] == 2:
            #     subset = 'validation'
            if is_test:
                subset = 'validation'
            else:
                subset = 'training'
            
            keys.append(row[0])
            subsets.append(subset)        
        if filename.split('.')[1] != 'json':
            for i in range(len(keys)):
                key = keys[i]
                database[key] = {}
                database[key]['subset'] = subsets[i]
                filename = filename.replace(" ", "_")
                label = filename.split('.')[0]
                #label = '_'.join(filename.split('_')[:-2])
                database[key]['annotations'] = {'label': label}
    
    return database

def get_labels(csv_dir_path):
    labels = []
    for name in os.listdir(csv_dir_path):
        print(name)
        if name.split('.')[1] != 'json':
            name = name.replace(" ", "_")
            labels.append(name.split('.')[0])
    print(labels)
    return sorted(list(set(labels)))

def convert_hmdb51_csv_to_activitynet_json(csv_dir_path, split_index, dst_json_path):
    labels = get_labels(csv_dir_path+"test")
    database1 = convert_csv_to_dict(csv_dir_path+"test", split_index,True)
    database2 = convert_csv_to_dict(csv_dir_path+"train", split_index,False)
    database = database1.copy()
    database.update(database2)
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__ == '__main__':
    csv_dir_path = sys.argv[1]

    for split_index in range(1, 2):
        dst_json_path = os.path.join(csv_dir_path, 'olympicSport_{}.json'.format(split_index))
        convert_hmdb51_csv_to_activitynet_json(csv_dir_path, split_index, dst_json_path)