# Enlist images in the annotated SUN09 database along withs patial relation labels
# Author: Akanksha Saran <asaran@cs.utexas.edu>
import os
import json

if __name__ == '__main__':
	data_folder = '/home/asaran/research/spatial-relations-attention/data/VG_100K/'
        training_img_num = 17189
        validation_img_num = 1000

	relation_dict = {
                'at': 0, 
                #'behind': 1,
                'across': 2,
                'near': 3,
                'next to': 3,
                'beside': 3,
                'around': 4,
                'above': 5,
                'on top of': 5,
                'side of': 6,
                'inside': 7,
                'in': 7,
                'over': 8,
                #'against': 9,
                'under': 10,
                'underneath': 10,
                'below': 10,
                'by': 11,
                'bottom': 12,
                'outside': 13,
                'on': 14,
                'right of': 15,
                'left of': 9,
                #'in front of': 17,
                'along': 1
                }

        with open('/home/asaran/research/spatial-relations-attention/data/relevant_relationships_bb_att.json') as relfile:
            qa = json.load(relfile)

        qa_train = qa[:training_img_num]
        qa_test = qa[training_img_num: training_img_num + validation_img_num]

        data_sets = [qa_train, qa_test]
        outfiles = ['../data/genome_train.json','../data/genome_test.json']

        
        for i in range(2):
            data = []
	    for img in data_sets[i]:
                img_id = img['image_id']
                for qas in img['relationships']:
                    if qas['answer'].lower() == 'yes':
                        relation = qas['predicate'].lower()
                        obj_bb = qas['object_bb']
                        subj_bb = qas['subject_bb']
                        height = qas['height']
                        width = qas['width']
                        # object = qas['object']
                        # subject = qas['subject']
                        bb_list = [int(obj_bb[0]/width), int(obj_bb[1]/height), int(obj_bb[2]/width), int(obj_bb[3]/height),\
                                int(subj_bb[0]/width), int(subj_bb[1]/height), int(subj_bb[2]/width), int(subj_bb[3]/height)]
                        
                        img_path = data_folder + str(img_id) + '.jpg'
                        for rel in relation_dict:
                            if rel in relation:
                                label = relation_dict[rel]
                                break

                        data_dict = {}
                        data_dict['img_path'] = img_path
                        data_dict['bb'] = bb_list
                        data_dict['label'] = label
                        data.append(data_dict)

	    with open(outfiles[i], 'w') as outfile:
                json.dump(data, outfile)


        

	

	
