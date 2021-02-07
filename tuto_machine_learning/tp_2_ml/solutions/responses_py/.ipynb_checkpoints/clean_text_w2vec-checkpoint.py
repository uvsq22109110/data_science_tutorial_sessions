df_finance = pd.read_csv("./fin_sentiment_analysis.csv",names=["sentiment","news"])

def get_harmonized_clean_text(text):
    
    #Stopwords list
    stopwords = nltk.corpus.stopwords.words('english') # removing stop words
    stopwords.remove("no")
    stopwords.remove("not")
    #Stemmer
    ps = PorterStemmer()
    clean_text = str(text).lower()
    clean_text = re.sub(r'http\S+', '', clean_text)
    clean_text = " ".join(re.findall("[a-z]+", clean_text))
    #stripping accent and punctuations
    clean_text = strip_accents(clean_text)
    clean_text = [elt for elt in clean_text.split() if elt not in stopwords]
    clean_text = [ps.stem(elt) for elt in clean_text]
    clean_text = [elt for elt in clean_text if elt not in stopwords]
    clean_text = list(set(clean_text))
    #Adding ngrams
    bigrams = [" ".join(elt) for elt in list(nltk.ngrams(clean_text,2))]
    trigrams = [" ".join(elt) for elt in list(nltk.ngrams(clean_text,3))]
    clean_text.extend(bigrams)
    clean_text.extend(trigrams)
        
    return clean_text

df_finance["news_cleaned"]= df_finance.news.apply(lambda x : get_harmonized_clean_text(x))
df_finance.head()