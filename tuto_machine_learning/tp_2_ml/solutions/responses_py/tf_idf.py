vectorizer = TfidfVectorizer(stop_words=english_stop_words,strip_accents="unicode",max_features=100)
x_train_embedding = vectorizer.fit_transform(X_train)
print(x_train_embedding.shape)
print(vectorizer.vocabulary_)
cos_similarities_regard_to_vector_0 = cosine_similarity(x_train_embedding[114],x_train_embedding)[0] # cosine  value
angle_similarities_regard_to_vector_0 = np.arccos(cos_similarities_regard_to_vector_0) #angle in radiant use np.degree to get the angle


fig, ax = plt.subplots(figsize=(10,10))
ax.arrow(0,0,np.cos(angle_similarities_regard_to_vector_0[114]),np.sin(angle_similarities_regard_to_vector_0[114]),head_width=0.01, head_length=0.01, fc='lightblue', ec='black')
ax.arrow(0,0,np.cos(angle_similarities_regard_to_vector_0[119]),np.sin(angle_similarities_regard_to_vector_0[119]),head_width=0.01, head_length=0.01, fc='lightblue', ec='blue')
ax.arrow(0,0,np.cos(angle_similarities_regard_to_vector_0[229]),np.sin(angle_similarities_regard_to_vector_0[229]),head_width=0.01, head_length=0.01, fc='lightblue', ec='orange')
ax.arrow(0,0,np.cos(angle_similarities_regard_to_vector_0[710]),np.sin(angle_similarities_regard_to_vector_0[710]),head_width=0.01, head_length=0.01, fc='lightblue', ec='red')
plt.show()

# Confirm using Cosine that similar sentences have cosine similarity near 1 and those different have a value near to zero and are orthogonal
for i in [114,119,229,710]:
    print("  Comment : ",list(X_train)[i],"     Cosine :",list(cos_similarities_regard_to_vector_0)[i])
