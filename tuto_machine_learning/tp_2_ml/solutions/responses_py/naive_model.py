def get_naive_pred_per_class(s):
    s = s.lower() #Setting string to lower
    s = strip_accents(s).split() #Generated Tokens of with a normal format
    s = [elt for elt in  s if(elt not in english_stop_words and len(elt)>1) ] # Cleaning data from stop words and removing words with legnth 1

    spam_likelihood_words = categories["spam"]["20_most_commun"]
    ham_likelihood_words = categories["ham"]["20_most_commun"]
    
    score_ham, score_spam = 0,0
    
    for word in s:
        if(word in spam_likelihood_words):
            score_spam+= (20-spam_likelihood_words.index(word)) # For the fist code just replace with 1
        elif(word in ham_likelihood_words):
            score_ham+= (20-ham_likelihood_words.index(word)) # For the fist code just replace with 1
            
    if(score_ham<score_spam):
        return "spam"
    elif(score_ham>score_spam):
        return 'ham'
    
    random.seed(42) #Reproductibility
    return  random.choice(["spam","ham"])
    
y_test_pred = le.transform([get_naive_pred_per_class(x) for x in X_test])
print(accuracy_score(y_test,y_test_pred)) 
#Barely greater than Random (50%) This model is better than ramdom but it as a bad model. We will use it as a baseline model