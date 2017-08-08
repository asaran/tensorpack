# Enlist images in the annotated SUN09 database along withs patial relation labels
# Author: Akanksha Saran <asaran@cs.utexas.edu>
import os

if __name__ == '__main__':
	train_folder = '../data/train_anno'
	test_folder = '../data/test_anno'

	categories = os.listdir(train_folder)
	relations = []
	rel_keys = {}
	rel_keys['in'] = -1
	category_keys = {}
	f = open('../data/relations.txt','w')
	for category in categories:
		relation = category.partition('-')[-1].rpartition('-')[0]
		category_keys[category] = relation
		if relation == 'inside_of':
			rel_keys[relation] = rel_keys['in']
			relation = 'in'
		if relation not in relations:
			rel_keys[relation] = len(relations)
			f.write(relation + '\t' + str(len(relations)) + '\n')
			relations.append(relation)
	f.close()
	
	f = open('../data/train.txt','w')
	for category in categories:
		img_list = os.listdir(train_folder + '/' + category)[0]
		imgs = [line.strip() for line in open(train_folder + '/' + category + '/' + img_list,'r')]
		for img in imgs:
			f.write('/home/asaran/research/tensorpack/examples/SimilarityLearning/data/static_sun09_database/' + img + '\t' + str(rel_keys[category_keys[category]]) + '\n')
	f.close()

	f = open('../data/test.txt','w')
	for category in categories:
		img_list = os.listdir(test_folder + '/' + category)[0]
		imgs = [line.strip() for line in open(test_folder + '/' + category + '/' + img_list,'r')]
		for img in imgs:
			f.write('/home/asaran/research/tensorpack/examples/SimilarityLearning/data/static_sun09_database/' + img + '\t' + str(rel_keys[category_keys[category]]) + '\n')
	f.close()
	
