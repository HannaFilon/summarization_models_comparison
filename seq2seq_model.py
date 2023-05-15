import seq2seq_functions as functions
import pandas as pd
import datasets
import nltk
import gensim.downloader as gensim_api
from tensorflow.keras import models, layers

dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')

lst_dics = [dic for dic in dataset["train"]]
dtf = pd.DataFrame(lst_dics).rename(columns={"article":"text", "highlights":"summary"})[["text","summary"]].head(20000)
dtf.to_csv("data_summary.csv", index=False)

lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords = lst_stopwords + ["cnn","say","said","new","wa","ha"]

dtf = functions.add_preprocessed_text(dtf, column="text",
                            punkt=True, lower=True, slang=True, lst_stopwords=lst_stopwords, lemm=True)
dtf = functions.add_preprocessed_text(dtf, column="summary",
                            punkt=True, lower=True, slang=True, lst_stopwords=lst_stopwords, lemm=True)


dtf_freq_text = functions.word_freq(corpus=dtf["text_clean"], ngrams=[1], top=30, figsize=(10,7))
dtf_freq_summary = functions.word_freq(corpus=dtf["summary_clean"], ngrams=[1], top=30, figsize=(10,7))
thres = 5
text_top_words = len(dtf_freq_text[dtf_freq_text["freq"]>thres])
summary_top_words = len(dtf_freq_summary[dtf_freq_text["freq"]>thres])
dtf = dtf[["text","text_clean","summary","summary_clean"]]
dtf_train = dtf.iloc[:16000]
dtf_test = dtf.iloc[16000:]

#Длины входной и выходной последовательности
max_len = 400
y_len = 40

dic_seq = functions.text2seq(corpus=dtf_train["text_clean"], top=text_top_words, maxlen=max_len)
X_train, X_tokenizer, X_dic_vocabulary = dic_seq["X"], dic_seq["tokenizer"], dic_seq["dic_vocabulary"]

X_test = functions.text2seq(corpus=dtf_test["text_clean"], fitted_tokenizer=X_tokenizer, maxlen=X_train.shape[1])

special_tokens = ("<START>", "<END>")
dtf_train["summary_clean"] = dtf_train['summary_clean'].apply(lambda x: special_tokens[0]+' '+x+' '+special_tokens[1])
dtf_test["summary_clean"] = dtf_test['summary_clean'].apply(lambda x: special_tokens[0]+' '+x+' '+special_tokens[1])

dic_seq = functions.text2seq(corpus=dtf_train["summary_clean"], top=summary_top_words, maxlen=y_len)
y_train, y_tokenizer, y_dic_vocabulary = dic_seq["X"], dic_seq["tokenizer"], dic_seq["dic_vocabulary"]
y_test = functions.text2seq(corpus=dtf_test["summary_clean"], fitted_tokenizer=y_tokenizer, maxlen=y_train.shape[1])
nlp = gensim_api.load("glove-wiki-gigaword-300")
word = "home"

nlp[word].shape
X_embeddings = functions.vocabulary_embeddings(X_dic_vocabulary, nlp)
X_embeddings.shape

y_embeddings = functions.vocabulary_embeddings(y_dic_vocabulary, nlp)
y_embeddings.shape


#Model Architecture
lstm_units = 250

x_in = layers.Input(name="x_in", shape=(X_train.shape[1],))
layer_x_emb = layers.Embedding(name="x_emb", input_dim=X_embeddings.shape[0], output_dim=X_embeddings.shape[1],
                               weights=[X_embeddings], trainable=False)
x_emb = layer_x_emb(x_in)
layer_x_bilstm = layers.Bidirectional(layers.LSTM(units=lstm_units, dropout=0.4, recurrent_dropout=0.4,
                                                  return_sequences=True, return_state=True),
                                      name="x_lstm_1")
x_out, _, _, _, _ = layer_x_bilstm(x_emb)
layer_x_bilstm = layers.Bidirectional(layers.LSTM(units=lstm_units, dropout=0.4, recurrent_dropout=0.4,
                                                  return_sequences=True, return_state=True),
                                      name="x_lstm_2")
x_out, _, _, _, _ = layer_x_bilstm(x_out)
layer_x_bilstm = layers.Bidirectional(layers.LSTM(units=lstm_units, dropout=0.4, recurrent_dropout=0.4,
                                                  return_sequences=True, return_state=True),
                                      name="x_lstm_3")
x_out, forward_h, forward_c, backward_h, backward_c = layer_x_bilstm(x_out)
state_h = layers.Concatenate()([forward_h, backward_h])
state_c = layers.Concatenate()([forward_c, backward_c])

y_in = layers.Input(name="y_in", shape=(None,))
layer_y_emb = layers.Embedding(name="y_emb", input_dim=y_embeddings.shape[0], output_dim=y_embeddings.shape[1],
                               weights=[y_embeddings], trainable=False)
y_emb = layer_y_emb(y_in)
layer_y_lstm = layers.LSTM(name="y_lstm", units=lstm_units*2, dropout=0.2, recurrent_dropout=0.2,
                           return_sequences=True, return_state=True)
y_out, _, _ = layer_y_lstm(y_emb, initial_state=[state_h, state_c])
layer_dense = layers.TimeDistributed(name="dense",
                                     layer=layers.Dense(units=len(y_dic_vocabulary), activation='softmax'))
y_out = layer_dense(y_out)

model = models.Model(inputs=[x_in, y_in], outputs=y_out, name="Seq2Seq")

optimizer='rmsprop'
loss_function='sparse_categorical_crossentropy'

model.compile(optimizer=optimizer, loss=loss_function)

print(model.summary())


num_epochs = 100
model,encoder_model,decoder_model = functions.fit_seq2seq(X_train, y_train, model, build_encoder_decoder=True,
                    epochs=num_epochs, batch_size=64, verbose=1)

model.save("Users/Maksim.Nevar/PycharmProjects/model_comparison/models/seq2seq_model")
encoder_model.save("Users/Maksim.Nevar/PycharmProjects/model_comparison/models/seq2seq_encoder_model")
decoder_model.save("Users/Maksim.Nevar/PycharmProjects/model_comparison/models/seq2seq_decoder_model")
predicted = functions.predict_seq2seq(X_test, encoder_model, decoder_model, y_tokenizer, special_tokens)

