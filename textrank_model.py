import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import networkx as nx
import sklearn.metrics.pairwise
import rouge
import datasets

dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[0:1%]")

dtf = pd.read_csv("data_summary.csv")

#lst_stopwords = nltk.corpus.stopwords.words("english")
#lst_stopwords = lst_stopwords + ["cnn", "say", "said", "new", "wa", "ha"]
#dtf = common_functions.add_preprocessed_text(dtf, column="text", punkt=False, lower=True, slang=True, lst_stopwords=lst_stopwords, lemm=True)
#dtf = common_functions.add_preprocessed_text(dtf, column="summary", punkt=False, lower=True, slang=True, lst_stopwords=lst_stopwords, lemm=True)
#dtf.to_csv("data_summary.csv", index=False)



dtf_test = dtf.iloc[16000:]
dtf_train = dtf.iloc[:16000]

text_index = 16002
print("---Text---")
print(dtf_test["text"][text_index])
print("---Summary---")
print(dtf_test["summary"][text_index])

embeddings_dict = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_dict[word] = coefs
f.close()

article=dtf_test["text"][text_index]
sentences = sent_tokenize(article)
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
clean_sentences = [s.lower() for s in clean_sentences]

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words = stop_words + ["cnn", "say", "said", "new", "wa", "ha"]

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

#сlean_sentences = [common_functions.final_clean_text(s) for s in clean_sentences]

sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([embeddings_dict.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)

sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = sklearn.metrics.pairwise.cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[
                    0, 0]

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

sn = 4 if len(ranked_sentences)>=4 else len(ranked_sentences)
predicted=""
for i in range(sn):
    predicted += ranked_sentences[i][1]

def evaluate_summary(summary_test, predicted):
    rouge_score = rouge.Rouge()
    scores = rouge_score.get_scores(summary_test, predicted, avg=True)
    score_1 = scores['rouge-1']['f']
    score_2 = scores['rouge-2']['f']
    score_L = scores['rouge-l']['f']

    print("---Original Summary---")
    print(summary_test)
    print("---Predicted Summary---")
    print(predicted)
       #print("---Rouge evaluation for predicted summary---")
        #print("Rouge1: ", score_1, "| Rouge2: ", score_2, "| RougeL: ",
            #score_L, "--> Avg Rouge:", round(np.mean([score_1,score_2,score_L]), 2))


evaluate_summary(dtf_test["summary"][text_index], predicted)


