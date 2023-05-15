import re
import contractions
import gensim.downloader as gensim_api
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras import callbacks, models, layers, preprocessing as kprocessing


def create_stopwords(lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
    lst_stopwords = set()
    for lang in lst_langs:
        lst_stopwords = lst_stopwords.union( set(nltk.corpus.stopwords.words(lang)) )
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))


def utils_preprocess_text(txt, lst_regex=None, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False,
                          lemm=True):
    ## Regex (in case, before cleaning)
    if lst_regex is not None:
        for regex in lst_regex:
            txt = re.sub(regex, '', txt)

    ## Clean
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
    ### slang
    txt = contractions.fix(txt) if slang is True else txt

    ## Tokenize (convert from string to list)
    lst_txt = txt.split()

    ## Stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]

    ## Lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]

    ## Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]

    ## Back to string
    txt = " ".join(lst_txt)
    return txt

def add_preprocessed_text(data, column, lst_regex=None, punkt=False, lower=False, slang=False, lst_stopwords=None,
                          stemm=False, lemm=False, remove_na=True):
    dtf = data.copy()

    ## apply preprocess
    dtf = dtf[pd.notnull(dtf[column])]
    dtf[column + "_clean"] = dtf[column].apply(
        lambda x: utils_preprocess_text(x, lst_regex, punkt, lower, slang, lst_stopwords, stemm, lemm))

    ## residuals
    dtf["check"] = dtf[column + "_clean"].apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column, column + "_clean"]][dtf["check"] == 0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"] > 0]

    return dtf.drop("check", axis=1)


def word_freq(corpus, ngrams=[1, 2, 3], top=10, figsize=(10, 7)):
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    ngrams = [ngrams] if type(ngrams) is int else ngrams

    ## calculate
    dtf_freq = pd.DataFrame()
    for n in ngrams:
        dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word", "freq"])
        dtf_n["ngrams"] = n
        dtf_freq = dtf_freq.append(dtf_n)
    dtf_freq["word"] = dtf_freq["word"].apply(lambda x: " ".join(string for string in x))
    dtf_freq = dtf_freq.sort_values(["ngrams", "freq"], ascending=[True, False])

    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_freq.groupby('ngrams')["ngrams", "freq", "word"].head(top))
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()
    return dtf_freq

def vocabulary_embeddings(dic_vocabulary, nlp=None):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    embeddings = np.zeros((len(dic_vocabulary)+1, nlp.vector_size))
    for word,idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] =  nlp[word]
        ## if word not in model then skip and the row stays all zeros
        except:
            pass
    print("vocabulary mapped to", embeddings.shape[0], "vectors of size", embeddings.shape[1])
    return embeddings


def utils_preprocess_ngrams(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[]):
    ## create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)

    ## detect common bi-grams and tri-grams
    if len(lst_ngrams_detectors) != 0:
        for detector in lst_ngrams_detectors:
            lst_corpus = list(detector[lst_corpus])
    return

def text2seq(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], fitted_tokenizer=None, top=None, oov=None,
             maxlen=None, padding="<PAD>"):
    print("--- tokenization ---")

    ## detect common n-grams in corpus
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join,
                                         lst_ngrams_detectors=lst_ngrams_detectors)

    ## bow with keras to get text2tokens without creating the sparse matrix
    ### train
    if fitted_tokenizer is None:
        tokenizer = kprocessing.text.Tokenizer(num_words=top, lower=False, split=' ', char_level=False, oov_token=oov,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(lst_corpus)
        dic_vocabulary = {padding: 0}
        words = tokenizer.word_index if top is None else dict(list(tokenizer.word_index.items())[0:top + 1])
        dic_vocabulary.update(words)
        print(len(dic_vocabulary), "words")
    else:
        tokenizer = fitted_tokenizer
    ### transform
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    ## padding sequence (from [1,2],[3,4,5,6] to [0,0,1,2],[3,4,5,6])
    print("--- padding to sequence ---")
    X = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=maxlen, padding="post", truncating="post")
    print(X.shape[0], "sequences of length", X.shape[1])

    ## plot heatmap
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(X == 0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sequences Overview')
    plt.show()
    return {"X": X, "tokenizer": tokenizer, "dic_vocabulary": dic_vocabulary} if fitted_tokenizer is None else X


def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()

    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()

def fit_seq2seq(X_train, y_train, model=None, X_embeddings=None, y_embeddings=None, build_encoder_decoder=True,
                epochs=100, batch_size=64, verbose=1):
    ## model
    if model is None:
        ### params
        len_vocabulary_X, embeddings_dim_X = X_embeddings.shape
        len_vocabulary_y, embeddings_dim_y = y_embeddings.shape
        lstm_units = 250
        max_seq_lenght = X_train.shape[1]
        ### encoder (embedding + lstm)
        x_in = layers.Input(name="x_in", shape=(max_seq_lenght,))
        layer_x_emb = layers.Embedding(name="x_emb", input_dim=len_vocabulary_X, output_dim=embeddings_dim_X,
                                       weights=[X_embeddings], trainable=False)
        x_emb = layer_x_emb(x_in)
        layer_x_lstm = layers.LSTM(name="x_lstm", units=lstm_units, return_sequences=True, return_state=True)
        x_out, state_h, state_c = layer_x_lstm(x_emb)
        ### decoder (embedding + lstm + dense)
        y_in = layers.Input(name="y_in", shape=(None,))
        layer_y_emb = layers.Embedding(name="y_emb", input_dim=len_vocabulary_y, output_dim=embeddings_dim_y,
                                       weights=[y_embeddings], trainable=False)
        y_emb = layer_y_emb(y_in)
        layer_y_lstm = layers.LSTM(name="y_lstm", units=lstm_units, return_sequences=True, return_state=True)
        y_out, _, _ = layer_y_lstm(y_emb, initial_state=[state_h, state_c])
        layer_dense = layers.TimeDistributed(name="dense",
                                             layer=layers.Dense(units=len_vocabulary_y, activation='softmax'))
        y_out = layer_dense(y_out)
        ### compile
        model = models.Model(inputs=[x_in, y_in], outputs=y_out, name="Seq2Seq")
        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())

    ## train
    training = model.fit(x=[X_train, y_train[:, :-1]],
                         y=y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
                         batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3,
                         callbacks=[callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)])
    if epochs > 1:
        utils_plot_keras_training(training)

    ## build prediction enconder-decoder model
    if build_encoder_decoder is True:
        lstm_units = lstm_units * 2 if any("Bidirectional" in str(layer) for layer in model.layers) else lstm_units
        ### encoder
        encoder_model = models.Model(inputs=x_in, outputs=[x_out, state_h, state_c], name="Prediction_Encoder")
        ### decoder
        encoder_out = layers.Input(shape=(max_seq_lenght, lstm_units))
        state_h, state_c = layers.Input(shape=(lstm_units,)), layers.Input(shape=(lstm_units,))
        y_emb2 = layer_y_emb(y_in)
        y_out2, new_state_h, new_state_c = layer_y_lstm(y_emb2, initial_state=[state_h, state_c])
        predicted_prob = layer_dense(y_out2)
        decoder_model = models.Model(inputs=[y_in, encoder_out, state_h, state_c],
                                     outputs=[predicted_prob, new_state_h, new_state_c],
                                     name="Prediction_Decoder")
        return training.model, encoder_model, decoder_model
    else:
        return training.model

