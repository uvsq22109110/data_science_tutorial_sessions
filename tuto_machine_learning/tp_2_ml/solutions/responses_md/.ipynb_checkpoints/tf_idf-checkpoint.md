1. TF and TF-IDF 
    - TF Terme Frequency : This method will compute the frequency of word of the corpus for each document
    - TF-IDF : TF stanf dor **term frequency** and IDF is the **inverse document frequency**
2. the formula of TF-IDF 
<div>
    <img src="https://www.seoquantum.com/sites/default/files/tf-idf-2-1-1024x375.png" /img>
</div>
3. Pros/Cons TF-IDF

    - Advantages:
    
        - Easy to compute
        - You have some basic metric to extract the most descriptive terms in a document
        - You can easily compute the similarity between 2 documents using it

    - Disadvantages:
        - TF-IDF is based on the bag-of-words (BoW) model, therefore it does not capture position in text, semantics, co-occurrences in different documents, etc.
        - For this reason, TF-IDF is only useful as a lexical level feature
        - Cannot capture semantics

4. As similarity function we can use the cosine distance. Cosine distance is the normalized dot product.
<div>
    <img src="https://clay-atlas.com/wp-content/uploads/2020/03/cosine-similarity.png" />
</div>
5. See code below
6. See code below
7. See code below
8. See code below