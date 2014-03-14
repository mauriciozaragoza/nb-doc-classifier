nb-doc-classifier
=================

This script classifies news documents among one of the classic 20 newsgroups dataset
http://qwone.com/~jason/20Newsgroups/

- To run this script it is necessary to have installed python 2.7.X and Scipy.

- The data will be retrieved from the folder data that the software assumes that
  is located at ../data/ with respect to this script.

- The files must be called as:
	test.data      -> Speciﬁes the counts for each of the words used in each of the documents for test
	test.label     -> Each line corresponds to the label for one document from the test set
	train.data     -> Each line corresponds to the label for one document from the training set
	train.label    -> Speciﬁes the counts for each of the words used in each of the documents for training
	vocabulary.txt -> A list of the words that may appear in the documents

- The script can accept one parameter from console, which will be an specific alpha, for example:
	python classifier.py 0.0003

- If no alpha is specified the script will use the default one: 1/|V|, where V is the vocabulary size,
  for example:
  	python classifier.py