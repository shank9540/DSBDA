import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data files (only once)
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample document
doc = "Natural Language Processing (NLP) helps computers understand human language and make sense of it."

# Step 1: Tokenization
tokens = word_tokenize(doc)
print("Tokens:", tokens)

# Step 2: POS Tagging
pos_tags = pos_tag(tokens)
print("\nPOS Tags:", pos_tags)

# Step 3: Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\nFiltered Tokens (No Stopwords & No Punctuation):", filtered_tokens)

# Step 4: Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed Tokens:", stemmed_tokens)

# Step 5: Lemmatization
lemmatizer = WordNetLemmatizer()

# Helper function to get wordnet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tag(filtered_tokens)]
print("\nLemmatized Tokens:", lemmatized_tokens)

# Step 6: TF-IDF Representation
documents = [doc]  # You can add more documents here
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

print("\nFeature Names:")
print(vectorizer.get_feature_names_out())
