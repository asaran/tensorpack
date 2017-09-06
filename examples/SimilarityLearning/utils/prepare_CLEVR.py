import json
import cv2

labels = {
            'left': 0,
            'right': 1,
            'front': 2,
            'behind': 3
        }

data_root = '/home/asaran/research/spatial-relations-attention/data/CLEVR_v1.0/'
with open(data_root+'scenes/CLEVR_train_scenes.json') as datafile:
    train_data = json.load(datafile)

with open(data_root+'scenes/CLEVR_val_scenes.json') as datafile:
    test_data = json.load(datafile)

train_data_list = []
test_data_list = []

for img in train_data['scenes']:
    objects = img['objects']
    #print('image_index: '+str(img['image_index']))
    #print('len(objects):' + str(len(objects)))
    #im = cv2.imread(data_root+'images/'+img['image_filename'])
    #width, height = 
    for relation in img['relationships']:
        for subj in range(len(img['relationships'][relation])):
            for obj in img['relationships'][relation][subj]:
                #print('obj: '+str(obj))
                #print('subj: '+str(subj))
                data_dict = {}
                data_dict['img_path'] = data_root+'images/train/'+img['image_filename']
                obj_pos = objects[obj]['pixel_coords']
                subj_pos = objects[subj]['pixel_coords']
                
                pos = obj_pos + subj_pos
                data_dict['position'] = pos
                data_dict['label'] = labels[relation]
                
                train_data_list.append(data_dict)

train_outfile = '/media/internal/asaran/CLEVR/train_relations.json'
with open(train_outfile, 'w') as outfile:
    json.dump(train_data_list, outfile)


for img in test_data['scenes']:
    objects = img['objects']
    #im = cv2.imread(data_root+'images/'+img['image_filename'])
    #width, height = 
    for relation in img['relationships']:
        for subj in range(len(img['relationships'][relation])):
            for obj in img['relationships'][relation][subj]:
                data_dict = {}
                data_dict['img_path'] = data_root+'images/val/'+img['image_filename']
                obj_pos = objects[obj]['pixel_coords']
                subj_pos = objects[subj]['pixel_coords']

                pos = obj_pos + subj_pos
                data_dict['position'] = pos
                data_dict['label'] = labels[relation]

                test_data_list.append(data_dict)

test_outfile = '/media/internal/asaran/CLEVR/test_relations.json'
with open(test_outfile, 'w') as outfile:
    json.dump(test_data_list, outfile)
