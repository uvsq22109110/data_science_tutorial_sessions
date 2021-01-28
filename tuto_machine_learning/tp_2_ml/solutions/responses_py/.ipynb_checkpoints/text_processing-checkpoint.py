df = pd.read_excel("./spam_ham.xls")
df["text_lower"] = df["text"].str.lower()
print(df["spam_ham"].value_counts())
categories = {}

f, axs = plt.subplots(1,2,figsize=(30,30))
ax_counter = 0
for cat in df["spam_ham"].unique():
    categories[cat] = {"text":df[df["spam_ham"]==cat]["text_lower"].values}
    categories[cat]["merged_text_lower"] = " ".join(categories[cat]["text"])
    wordcloud = WordCloud(max_font_size=50, max_words=300, random_state=42, background_color="white").generate(categories[cat]["merged_text_lower"] )
    axs[ax_counter].imshow(wordcloud, interpolation="bilinear")
    axs[ax_counter].axis("off")
    axs[ax_counter].set_title(cat)
    ax_counter +=1
plt.show()




english_stop_words = stopwords.words('english')
words_to_add_for_example = ["u","said", "ì_","ì", 'lt',"gt","late","ur","d","yes","hi","å","l","t","g","u","d","u","&lt;#&gt;","ok","got","i'd",'get', 'go', 'know',"call"]
english_stop_words.extend(words_to_add_for_example)

def strip_accents(s):
    
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    pattern = "["+string.punctuation.replace(".","").replace("@","")+"]"

    s = _RE_COMBINE_WHITESPACE.sub(" ", re.sub(pattern, " ", s) ).strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

print("After Cleaning Data")

f, axs = plt.subplots(1,2,figsize=(30,30))
ax_counter = 0
for cat in df["spam_ham"].unique(): 
    # Easier way is to take all the string concatenate them them merge ;) (And not to split thne merge)
    categories[cat]["tokenized_lower_text"] = strip_accents(categories[cat]["merged_text_lower"]).split(" ")
    categories[cat]["harmonized_cleaned_text"] = " ".join([elt for elt in categories[cat]["tokenized_lower_text"] if(elt not in english_stop_words and len(elt)>1)])
    wordcloud = WordCloud(max_font_size=50, max_words=300, random_state=42, background_color="white").generate(categories[cat]["harmonized_cleaned_text"])
    axs[ax_counter].imshow(wordcloud, interpolation="bilinear")
    axs[ax_counter].axis("off")
    axs[ax_counter].set_title(cat)
    ax_counter +=1
plt.show()

for cat in df["spam_ham"].unique(): 
    categories[cat]["20_most_commun"] = [elt[0] for elt in collections.Counter(categories[cat]["harmonized_cleaned_text"].split()).most_common(20)]
    print(cat, categories[cat]["20_most_commun"])
