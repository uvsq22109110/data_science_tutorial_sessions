w2v_model = gensim.models.Word2Vec(df_finance.news_cleaned.values,min_count=2,window=5,size=100,iter=1000,sg=1,workers=1,seed=42)
vocabulary = list(w2v_model.wv.vocab)

print(w2v_model.most_similar(positive=['madonna', 'ibrahimovich'], negative=['swift'], topn=1))
print(w2v_model.most_similar(positive=['ronaldo']))

def to_vector_space(model,vocab, clean_sentence):
    
    clean_sentence = list(set(clean_sentence))
    bigrams = [" ".join(elt) for elt in list(nltk.ngrams(clean_sentence,2))]
    trigrams = [" ".join(elt) for elt in list(nltk.ngrams(clean_sentence,3))]
    clean_sentence.extend(bigrams)
    clean_sentence.extend(trigrams)
    tokens = [x for x in clean_sentence if x in vocab]
    if(len(tokens)):
        return np.mean(w2v_model.__getitem__(tokens),0)
    return np.zeros(100)
df_finance["news_vectors"] = df_finance["news_cleaned"].apply(lambda x : to_vector_space(w2v_model, vocabulary, x))
df_finance.head()

max_vocab = 100
pca = PCA(n_components=2,random_state=42)
pca_vocabulary_vectors = pca.fit_transform(w2v_model[vocabulary])
print(pca.explained_variance_ratio_)
# create a scatter plot of the projection
plt.figure(figsize=(25,20))
plt.scatter(pca_vocabulary_vectors[:max_vocab, 0], pca_vocabulary_vectors[:max_vocab, 1])
for i, word in enumerate(vocabulary):
    plt.text(pca_vocabulary_vectors[i, 0], pca_vocabulary_vectors[i, 1],word,fontsize=14)
    if(i > max_vocab):
        break
plt.show()


df_finance.sentiment.value_counts().plot(kind='bar')
plt.show()
