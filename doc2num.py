# coding=utf-8

import ExFileManager
import re
import gensim
from pprint import pprint
from gensim import corpora, matutils, models
from collections import defaultdict
import tqdm


ex_file_manager = ExFileManager.ExFileManager()


def split_each_review(doc):
	"""
	long str will be split to 2d-list
	"""

	regex = re.compile(r"[0-9]\t.*")
	splited_doc = re.findall(regex,doc)
	return splited_doc

def remove_stop_word(doc):

	stop_words = set('at out from all for a of the and to in ( ) . it is are be I you he fils movie she - -- , who was has have --- this on that his her by . i one an " as with'.split())
	doc = [[word for word in d.lower().split() if word not in stop_words] for d in doc]
	return doc


def clean_doc(doc):

	reviews = []
	for i, d in enumerate(doc):
		d = re.sub('\t'," ", d)
		doc[i] = d
		reviews.append(int(d[0]))
	return doc, reviews


def TF_IDF_Reg(corpus_list):

	tfidf_model = models.TfidfModel(corpus_list, normalize=True)
	courpus_list_tfidf = tfidf_model[corpus_list]
	word_matrix = matutils.corpus2csc(courpus_list_tfidf)

	return word_matrix


path_to_test_data = "review_polarity_test"
path_to_train_data = "review_polarity_train"

#############
# Preproces #
#############

# [1] Reading of Data
# Reads txt type document data as a long string
long_string = ex_file_manager.read_txt_file(path_to_train_data)

# [2] Cleaning of Data

doc = split_each_review(long_string)
doc, reviews = clean_doc(doc)
doc = remove_stop_word(doc)

frequency = defaultdict(int)
for d in doc:
	for token in d:
		frequency[token] += 1

doc = [[token for token in d if frequency[token] > 1] for d in doc]

dictionary = corpora.Dictionary(doc)

corpus = [dictionary.doc2bow(d) for d in doc]


####################
# Building a Model #
####################

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=2, id2word=dictionary)
pprint(lda.show_topics())


# 文を定義

test_doc = ex_file_manager.read_txt_file(path_to_test_data)

# [2] Cleaning of Data
test_doc = split_each_review(test_doc)
test_doc,test_reviews = clean_doc(test_doc)
test_doc = remove_stop_word(test_doc)

test_doc = test_doc[0:14]

# 既存の辞書を使用して、コーパスを作成
test_corpus = [dictionary.doc2bow(text) for text in test_doc]


for topics_per_document in lda[test_corpus]:
	pprint(topics_per_document)

	p0 = topics_per_document[0][1]
	p1 = topics_per_document[0][1]

	if (p0 > p1):
		print("0")
	else:
		print("1")
