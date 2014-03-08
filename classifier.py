nY = 20
nX = 61188

cY = [0 for i in range(nY)]
doc_label = [0 for i in range(nX)]

bag = [[0 for i in range(nX)] for j in range(nY)]

def read_counts(filename):
	document = 0

	for line in open(filename, "r"):
		label = int(line.strip()) - 1
		cY[label] += 1
		doc_label[document] = label
		document += 1

def read_bag(filename):
	for line in open(filename, "r"):
		params = line.split(" ");
		doc_id = int(params[0])
		word_id = int(params[1]) 
		word_count = int(params[2].strip())

		label = doc_label[doc_id]

		bag[label][word_id] += word_count

# MAIN

read_counts("train.label")
read_bag("train.data")