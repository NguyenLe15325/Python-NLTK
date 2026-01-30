# NLTK (Natural Language Toolkit) - Complete Comprehensive Guide
## From Fundamentals to Advanced

---

## Table of Contents

1. [Introduction & Setup](#1-introduction--setup)
2. [Text Processing Fundamentals](#2-text-processing-fundamentals)
3. [Tokenization](#3-tokenization)
4. [Stopwords & Text Cleaning](#4-stopwords--text-cleaning)
5. [Stemming](#5-stemming)
6. [Lemmatization](#6-lemmatization)
7. [Part-of-Speech (POS) Tagging](#7-part-of-speech-pos-tagging)
8. [Named Entity Recognition (NER)](#8-named-entity-recognition-ner)
9. [Chunking & Parsing](#9-chunking--parsing)
10. [N-Grams](#10-n-grams)
11. [Frequency Distribution & Collocations](#11-frequency-distribution--collocations)
12. [WordNet & Semantic Analysis](#12-wordnet--semantic-analysis)
13. [Sentiment Analysis](#13-sentiment-analysis)
14. [Text Classification](#14-text-classification)
15. [Corpus Management](#15-corpus-management)
16. [Advanced Topics](#16-advanced-topics)
17. [Real-World Projects](#17-real-world-projects)

---

## 1. Introduction & Setup

### 1.1 What is NLTK?
NLTK (Natural Language Toolkit) is a leading platform for building Python programs to work with human language data. It provides:
- Easy-to-use interfaces to over 50 corpora and lexical resources
- Text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning
- Wrappers for industrial-strength NLP libraries

### 1.2 Installation

```python
# Install NLTK
pip install nltk

# Install with all dependencies
pip install nltk numpy scipy matplotlib
```

### 1.3 Downloading NLTK Data

```python
import nltk

# Download all data (recommended for beginners)
nltk.download('all')

# Download specific packages
nltk.download('punkt')          # Tokenizers
nltk.download('stopwords')      # Stop words
nltk.download('wordnet')        # WordNet lexical database
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('maxent_ne_chunker')  # Named Entity chunker
nltk.download('words')          # Word list
nltk.download('vader_lexicon')  # Sentiment analysis

# Interactive downloader
nltk.download()  # Opens GUI
```

### 1.4 Verifying Installation

```python
import nltk
print(nltk.__version__)

# Test basic functionality
from nltk.tokenize import word_tokenize
text = "Hello, NLTK is working!"
print(word_tokenize(text))
```

---

## 2. Text Processing Fundamentals

### 2.1 Working with Text

```python
import nltk

# Raw text
text = """Natural Language Processing (NLP) is a field of artificial intelligence 
that gives computers the ability to understand text and spoken words."""

# Basic string operations
print(f"Length: {len(text)}")
print(f"Uppercase: {text.upper()}")
print(f"Lowercase: {text.lower()}")
```

### 2.2 NLTK Text Object

```python
from nltk.text import Text
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
nltk_text = Text(tokens)

# Concordance - shows word in context
nltk_text.concordance("processing")

# Similar words
nltk_text.similar("processing")

# Common contexts
nltk_text.common_contexts(["language", "text"])

# Dispersion plot (requires matplotlib)
nltk_text.dispersion_plot(["language", "processing", "computers"])
```

### 2.3 Loading Sample Texts

```python
from nltk.book import *

# Pre-loaded texts
# text1: Moby Dick
# text2: Sense and Sensibility
# text3: Book of Genesis
# text4: Inaugural Address Corpus
# text5: Chat Corpus
# text6: Monty Python
# text7: Wall Street Journal
# text8: Personals Corpus
# text9: The Man Who Was Thursday

text1.concordance("whale")
text4.dispersion_plot(["citizens", "democracy", "freedom", "America"])
```

---

## 3. Tokenization

### 3.1 Word Tokenization

```python
from nltk.tokenize import word_tokenize, wordpunct_tokenize, WhitespaceTokenizer

text = "Hello! How are you? I'm doing fine, thanks."

# Standard word tokenizer
tokens = word_tokenize(text)
print(tokens)
# ['Hello', '!', 'How', 'are', 'you', '?', 'I', "'m", 'doing', 'fine', ',', 'thanks', '.']

# Punctuation-based tokenizer
punct_tokens = wordpunct_tokenize(text)
print(punct_tokens)
# ['Hello', '!', 'How', 'are', 'you', '?', 'I', "'", 'm', 'doing', 'fine', ',', 'thanks', '.']

# Whitespace tokenizer
ws_tokenizer = WhitespaceTokenizer()
ws_tokens = ws_tokenizer.tokenize(text)
print(ws_tokens)
```

### 3.2 Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer

text = """Dr. Smith went to the store. He bought milk. 
The price was $3.50! Can you believe it?"""

# Standard sentence tokenizer
sentences = sent_tokenize(text)
for i, sent in enumerate(sentences):
    print(f"{i+1}: {sent}")

# Custom trained sentence tokenizer
custom_tokenizer = PunktSentenceTokenizer(text)
custom_sentences = custom_tokenizer.tokenize(text)
```

### 3.3 Regular Expression Tokenizer

```python
from nltk.tokenize import RegexpTokenizer, regexp_tokenize

text = "Hello! Email: user@example.com, Phone: 123-456-7890"

# Tokenize only words (no punctuation)
word_tokenizer = RegexpTokenizer(r'\w+')
words = word_tokenizer.tokenize(text)
print(words)
# ['Hello', 'Email', 'user', 'example', 'com', 'Phone', '123', '456', '7890']

# Tokenize email addresses
email_pattern = r'[\w\.-]+@[\w\.-]+'
emails = regexp_tokenize(text, email_pattern)
print(emails)
# ['user@example.com']

# Tokenize phone numbers
phone_pattern = r'\d{3}-\d{3}-\d{4}'
phones = regexp_tokenize(text, phone_pattern)
print(phones)
# ['123-456-7890']
```

### 3.4 Tweet Tokenizer

```python
from nltk.tokenize import TweetTokenizer

tweet = "This is a cooool #NLP tweet! @user :-) 😊 https://example.com"

# Standard tokenizer handles emojis/hashtags poorly
tknzr = TweetTokenizer(
    preserve_case=False,      # Lowercase
    strip_handles=True,       # Remove @mentions
    reduce_len=True           # Reduce repeated chars
)

tokens = tknzr.tokenize(tweet)
print(tokens)
# ['this', 'is', 'a', 'cool', '#nlp', 'tweet', '!', ':-)', '😊', 'https://example.com']
```

### 3.5 Multi-Word Expression Tokenizer

```python
from nltk.tokenize import MWETokenizer

tokenizer = MWETokenizer()

# Add multi-word expressions
tokenizer.add_mwe(('New', 'York'))
tokenizer.add_mwe(('United', 'States'))
tokenizer.add_mwe(('machine', 'learning'))

text = word_tokenize("I love machine learning in New York")
mwe_tokens = tokenizer.tokenize(text)
print(mwe_tokens)
# ['I', 'love', 'machine_learning', 'in', 'New_York']
```

---

## 4. Stopwords & Text Cleaning

### 4.1 Stopwords

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Get English stopwords
stop_words = set(stopwords.words('english'))
print(f"Number of stopwords: {len(stop_words)}")
print(list(stop_words)[:20])

# Available languages
print(stopwords.fileids())

# Remove stopwords from text
text = "This is a sample sentence showing off the stop words filtration."
tokens = word_tokenize(text.lower())
filtered = [w for w in tokens if w not in stop_words and w.isalnum()]
print(f"Original: {tokens}")
print(f"Filtered: {filtered}")
```

### 4.2 Custom Stopwords

```python
# Extend default stopwords
custom_stopwords = stop_words.union({'also', 'however', 'therefore'})

# Create domain-specific stopwords
medical_stopwords = stop_words.union({'patient', 'doctor', 'hospital', 'treatment'})

# Remove certain stopwords
modified_stopwords = stop_words - {'not', 'no', 'nor'}  # Keep negations
```

### 4.3 Complete Text Cleaning Pipeline

```python
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    """Complete text cleaning pipeline"""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Tokenize
    tokens = word_tokenize(text)
    
    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # 7. Remove short tokens
    tokens = [t for t in tokens if len(t) > 2]
    
    return tokens

# Example
dirty_text = "Check out https://example.com! <b>Amazing</b> NLP tutorial #1!!!"
clean = clean_text(dirty_text)
print(clean)
```

---

## 5. Stemming

### 5.1 Porter Stemmer

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()

words = ["running", "runs", "runner", "ran", "easily", "fairly", 
         "studies", "studying", "studied", "connection", "connected"]

for word in words:
    print(f"{word} -> {ps.stem(word)}")

# Output:
# running -> run
# runs -> run
# runner -> runner
# ran -> ran
# easily -> easili
# fairly -> fairli
# studies -> studi
# studying -> studi
# studied -> studi
# connection -> connect
# connected -> connect
```

### 5.2 Lancaster Stemmer (More Aggressive)

```python
from nltk.stem import LancasterStemmer

ls = LancasterStemmer()

words = ["running", "maximum", "presumably", "multiply"]

for word in words:
    porter = ps.stem(word)
    lancaster = ls.stem(word)
    print(f"{word}: Porter={porter}, Lancaster={lancaster}")

# Lancaster is more aggressive, often produces shorter stems
```

### 5.3 Snowball Stemmer (Multi-language)

```python
from nltk.stem import SnowballStemmer

# Available languages
print(SnowballStemmer.languages)

# English
english_stemmer = SnowballStemmer("english")
print(english_stemmer.stem("generously"))  # generous

# Spanish
spanish_stemmer = SnowballStemmer("spanish")
print(spanish_stemmer.stem("corriendo"))  # corr

# German
german_stemmer = SnowballStemmer("german")
print(german_stemmer.stem("laufenden"))  # lauf
```

### 5.4 Regexp Stemmer

```python
from nltk.stem import RegexpStemmer

# Custom suffix removal
reg_stemmer = RegexpStemmer('ing$|ed$|s$|able$', min=4)

words = ["running", "walked", "cats", "readable"]
for word in words:
    print(f"{word} -> {reg_stemmer.stem(word)}")
```

### 5.5 Comparing Stemmers

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer("english")

test_words = ["programming", "programmer", "programmed", "programs",
              "generalization", "generalizing", "generalized"]

print(f"{'Word':<20} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 65)

for word in test_words:
    print(f"{word:<20} {ps.stem(word):<15} {ls.stem(word):<15} {ss.stem(word):<15}")
```

---

## 6. Lemmatization

### 6.1 WordNet Lemmatizer

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# Default (noun)
print(lemmatizer.lemmatize("cats"))       # cat
print(lemmatizer.lemmatize("running"))    # running (wrong without POS)

# With POS tag
print(lemmatizer.lemmatize("running", pos='v'))  # run
print(lemmatizer.lemmatize("better", pos='a'))   # good
print(lemmatizer.lemmatize("quickly", pos='r'))  # quickly
```

### 6.2 POS Tag Mapping for Lemmatization

```python
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    """Convert TreeBank POS tag to WordNet POS tag"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    lemmas = []
    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word.lower(), wn_tag)
        lemmas.append(lemma)
    
    return lemmas

# Example
sentence = "The striped bats are hanging on their feet for best"
print(lemmatize_sentence(sentence))
# ['the', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'good']
```

### 6.3 Stemming vs Lemmatization Comparison

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = [
    ("better", 'a'),
    ("running", 'v'),
    ("studies", 'n'),
    ("feet", 'n'),
    ("wolves", 'n'),
    ("caring", 'v')
]

print(f"{'Word':<15} {'Stemmed':<15} {'Lemmatized':<15}")
print("-" * 45)

for word, pos in words:
    stemmed = ps.stem(word)
    lemmatized = lemmatizer.lemmatize(word, pos)
    print(f"{word:<15} {stemmed:<15} {lemmatized:<15}")
```

---

## 7. Part-of-Speech (POS) Tagging

### 7.1 Basic POS Tagging

```python
from nltk import pos_tag, word_tokenize

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

print(tagged)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), 
#  ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]
```

### 7.2 POS Tag Meanings

```python
import nltk

# Get help on POS tags
nltk.help.upenn_tagset()

# Look up specific tag
nltk.help.upenn_tagset('VBZ')

# Common POS Tags:
# NN    - Noun, singular
# NNS   - Noun, plural
# NNP   - Proper noun, singular
# NNPS  - Proper noun, plural
# VB    - Verb, base form
# VBD   - Verb, past tense
# VBG   - Verb, gerund/present participle
# VBN   - Verb, past participle
# VBP   - Verb, non-3rd person singular present
# VBZ   - Verb, 3rd person singular present
# JJ    - Adjective
# JJR   - Adjective, comparative
# JJS   - Adjective, superlative
# RB    - Adverb
# RBR   - Adverb, comparative
# RBS   - Adverb, superlative
# DT    - Determiner
# IN    - Preposition or subordinating conjunction
# CC    - Coordinating conjunction
# PRP   - Personal pronoun
# PRP$  - Possessive pronoun
```

### 7.3 Extracting Specific POS

```python
from nltk import pos_tag, word_tokenize

text = """Natural language processing enables computers to understand, 
interpret, and generate human language. Machine learning algorithms 
help improve these capabilities over time."""

tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# Extract nouns
nouns = [word for word, tag in tagged if tag.startswith('NN')]
print(f"Nouns: {nouns}")

# Extract verbs
verbs = [word for word, tag in tagged if tag.startswith('VB')]
print(f"Verbs: {verbs}")

# Extract adjectives
adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
print(f"Adjectives: {adjectives}")

# Extract all with specific tags
def extract_by_pos(tagged_tokens, pos_prefix):
    return [(word, tag) for word, tag in tagged_tokens 
            if tag.startswith(pos_prefix)]
```

### 7.4 Custom POS Tagger

```python
from nltk import UnigramTagger, BigramTagger, TrigramTagger
from nltk.corpus import brown

# Training data
train_sents = brown.tagged_sents(categories='news')[:3000]
test_sents = brown.tagged_sents(categories='news')[3000:3500]

# Unigram tagger (based on single words)
unigram_tagger = UnigramTagger(train_sents)
print(f"Unigram accuracy: {unigram_tagger.accuracy(test_sents):.2%}")

# Bigram tagger with backoff to unigram
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
print(f"Bigram accuracy: {bigram_tagger.accuracy(test_sents):.2%}")

# Trigram tagger with backoff chain
trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger)
print(f"Trigram accuracy: {trigram_tagger.accuracy(test_sents):.2%}")
```

### 7.5 Brill Tagger (Transformation-Based)

```python
from nltk.tag import brill, brill_trainer
from nltk.tagger import DefaultTagger, UnigramTagger

# Base taggers
default_tagger = DefaultTagger('NN')
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)

# Brill tagger templates
templates = brill.nltkdemo18()

# Train Brill tagger
trainer = brill_trainer.BrillTaggerTrainer(
    unigram_tagger, templates, trace=3
)
brill_tagger = trainer.train(train_sents, max_rules=200)

print(f"Brill accuracy: {brill_tagger.accuracy(test_sents):.2%}")
```

---

## 8. Named Entity Recognition (NER)

### 8.1 Basic NER

```python
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

text = "Apple Inc. was founded by Steve Jobs in California. Tim Cook is the current CEO."

# Tokenize and POS tag
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# Named Entity Recognition
tree = ne_chunk(tagged)
print(tree)

# Extract named entities
def extract_named_entities(tree):
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity = " ".join([word for word, tag in subtree])
            entity_type = subtree.label()
            entities.append((entity, entity_type))
    return entities

entities = extract_named_entities(tree)
for entity, entity_type in entities:
    print(f"{entity}: {entity_type}")
```

### 8.2 NER Entity Types

```python
# NLTK NER recognizes these entity types:
# PERSON      - People's names
# ORGANIZATION- Companies, agencies, institutions
# GPE         - Geo-Political Entity (countries, cities, states)
# LOCATION    - Non-GPE locations (mountains, bodies of water)
# FACILITY    - Buildings, airports, highways
# GSP         - Geo-Socio-Political group
```

### 8.3 Binary NER (Named Entity vs Not)

```python
# Binary mode - just identifies if something is a NE or not
tree_binary = ne_chunk(tagged, binary=True)

def extract_ne_binary(tree):
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label') and subtree.label() == 'NE':
            entity = " ".join([word for word, tag in subtree])
            entities.append(entity)
    return entities

print(extract_ne_binary(tree_binary))
```

### 8.4 Visualizing NER Trees

```python
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.draw import TreeView

text = "Barack Obama was born in Hawaii and served as President of United States."
tree = ne_chunk(pos_tag(word_tokenize(text)))

# Display tree (opens GUI)
# tree.draw()

# Print tree structure
tree.pprint()

# Convert to IOB format
iob_tags = nltk.tree2conlltags(tree)
for word, pos, ne in iob_tags:
    print(f"{word:<15} {pos:<8} {ne}")
```

### 8.5 NER with Stanford NER (Advanced)

```python
# Note: Requires Stanford NER JAR files
from nltk.tag import StanfordNERTagger

# Set paths (adjust to your installation)
# stanford_ner_jar = '/path/to/stanford-ner.jar'
# stanford_model = '/path/to/english.all.3class.distsim.crf.ser.gz'

# st = StanfordNERTagger(stanford_model, stanford_ner_jar)

# text = "Google was founded by Larry Page and Sergey Brin in California"
# tokenized = word_tokenize(text)
# classified = st.tag(tokenized)
# print(classified)
```

---

## 9. Chunking & Parsing

### 9.1 Noun Phrase Chunking

```python
import nltk
from nltk import word_tokenize, pos_tag, RegexpParser

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# Define chunk grammar
grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}     # Noun Phrase
    VP: {<VB.*><NP|PP|CLAUSE>+} # Verb Phrase
    PP: {<IN><NP>}              # Prepositional Phrase
"""

chunk_parser = RegexpParser(grammar)
tree = chunk_parser.parse(tagged)

print(tree)
# tree.draw()  # Visualize
```

### 9.2 Custom Chunking Patterns

```python
# Named Entity Pattern
grammar_ne = r"""
    NAME: {<NNP>+}
    PLACE: {<DT>?<NNP>+<NN>?}
"""

# Technical Terms
grammar_tech = r"""
    TECH: {<JJ>*<NN><NN>*}     # Compound nouns
    ACRO: {<NNP><NNP>+}        # Acronyms/abbreviations
"""

# Verb Groups
grammar_vp = r"""
    VP: {<RB>?<VB.*>+<RB>?}    # Adverb + Verb + Adverb
"""

chunk_parser = RegexpParser(grammar_tech)
text = "The natural language processing system uses machine learning"
tagged = pos_tag(word_tokenize(text))
tree = chunk_parser.parse(tagged)
print(tree)
```

### 9.3 Chinking (Excluding from Chunks)

```python
# Chinking: Defining what to EXCLUDE from chunks
grammar = r"""
    NP: {<.*>+}         # Chunk everything
        }<VB.*|IN>+{    # Chink verbs and prepositions
"""

text = "The cat sat on the mat"
tagged = pos_tag(word_tokenize(text))
chunk_parser = RegexpParser(grammar)
tree = chunk_parser.parse(tagged)
print(tree)
```

### 9.4 Context-Free Grammar (CFG) Parsing

```python
from nltk import CFG
from nltk.parse import RecursiveDescentParser, ChartParser

# Define grammar
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> DT NN | DT JJ NN | NP PP
    VP -> VB NP | VP PP
    PP -> IN NP
    DT -> 'the' | 'a'
    NN -> 'dog' | 'cat' | 'park' | 'mat'
    JJ -> 'big' | 'small' | 'lazy'
    VB -> 'chased' | 'sat' | 'saw'
    IN -> 'on' | 'in' | 'at'
""")

# Create parser
rd_parser = RecursiveDescentParser(grammar)
chart_parser = ChartParser(grammar)

# Parse sentence
sentence = ['the', 'dog', 'chased', 'a', 'cat']
for tree in rd_parser.parse(sentence):
    print(tree)
    # tree.draw()
```

### 9.5 Dependency Parsing

```python
# NLTK has limited dependency parsing
# For production, consider spaCy or Stanford Parser

from nltk.parse.dependencygraph import DependencyGraph

# Manual dependency representation
dep_str = """
The     DT      2       det
quick   JJ      4       amod
brown   JJ      4       amod
fox     NN      5       nsubj
jumps   VBZ     0       ROOT
over    IN      8       case
the     DT      8       det
dog     NN      5       obl
"""

# dg = DependencyGraph(dep_str)
# print(dg.tree())
```

---

## 10. N-Grams

### 10.1 Creating N-Grams

```python
from nltk import ngrams, bigrams, trigrams
from nltk.tokenize import word_tokenize

text = "Natural language processing is very interesting"
tokens = word_tokenize(text.lower())

# Bigrams (n=2)
bi = list(bigrams(tokens))
print(f"Bigrams: {bi}")

# Trigrams (n=3)
tri = list(trigrams(tokens))
print(f"Trigrams: {tri}")

# General n-grams
four_grams = list(ngrams(tokens, 4))
print(f"4-grams: {four_grams}")

# With padding
padded_bi = list(ngrams(tokens, 2, pad_left=True, pad_right=True,
                         left_pad_symbol='<s>', right_pad_symbol='</s>'))
print(f"Padded bigrams: {padded_bi}")
```

### 10.2 Character N-Grams

```python
from nltk import ngrams

word = "language"

# Character bigrams
char_bigrams = list(ngrams(word, 2))
print(f"Character bigrams: {char_bigrams}")
# [('l', 'a'), ('a', 'n'), ('n', 'g'), ('u', 'a'), ('a', 'g'), ('g', 'e')]

# Character trigrams
char_trigrams = [''.join(g) for g in ngrams(word, 3)]
print(f"Character trigrams: {char_trigrams}")
```

### 10.3 N-Gram Language Model

```python
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize, sent_tokenize

# Training corpus
corpus = """
Natural language processing is a field of computer science.
Machine learning enables computers to learn from data.
Deep learning is a subset of machine learning.
"""

# Prepare data
sentences = sent_tokenize(corpus)
tokenized = [word_tokenize(s.lower()) for s in sentences]

# Create training data for trigrams (n=3)
n = 3
train_data, vocab = padded_everygram_pipeline(n, tokenized)

# Train MLE model
model = MLE(n)
model.fit(train_data, vocab)

# Generate text
print(model.generate(10, text_seed=['natural', 'language']))

# Get probability
print(model.score('processing', ['natural', 'language']))
```

### 10.4 N-Gram Frequency Analysis

```python
from nltk import FreqDist, bigrams
from nltk.tokenize import word_tokenize

text = """
The cat sat on the mat. The cat was very happy. 
The mat was soft. The cat slept on the mat.
"""

tokens = word_tokenize(text.lower())

# Bigram frequency
bi = list(bigrams(tokens))
bi_freq = FreqDist(bi)

print("Most common bigrams:")
for gram, freq in bi_freq.most_common(5):
    print(f"  {gram}: {freq}")
```

---

## 11. Frequency Distribution & Collocations

### 11.1 Frequency Distribution

```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = """
Natural language processing is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers 
and human language. Natural language processing has many applications.
"""

# Tokenize and clean
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
words = [w for w in tokens if w.isalpha() and w not in stop_words]

# Create frequency distribution
fdist = FreqDist(words)

# Most common words
print("Most common words:")
print(fdist.most_common(10))

# Frequency of specific word
print(f"\nFrequency of 'language': {fdist['language']}")

# Number of unique words
print(f"Unique words: {len(fdist)}")

# Words appearing once (hapaxes)
print(f"Hapaxes: {fdist.hapaxes()}")

# Plot frequency distribution
# fdist.plot(20, cumulative=False)
```

### 11.2 Conditional Frequency Distribution

```python
from nltk import ConditionalFreqDist
from nltk.corpus import brown

# Word frequency by genre
cfd = ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

# Compare word usage across genres
genres = ['news', 'romance', 'science_fiction']
words = ['love', 'money', 'science', 'technology']

for word in words:
    print(f"\n'{word}' frequency:")
    for genre in genres:
        print(f"  {genre}: {cfd[genre][word]}")

# Plot
# cfd.tabulate(conditions=genres, samples=words)
# cfd.plot(conditions=genres, samples=words)
```

### 11.3 Collocations

```python
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.tokenize import word_tokenize
from nltk.corpus import webtext

# Sample text
text = webtext.raw('firefox.txt')
tokens = word_tokenize(text.lower())

# Bigram collocations
bigram_finder = BigramCollocationFinder.from_words(tokens)

# Apply frequency filter
bigram_finder.apply_freq_filter(3)

# Different association measures
bigram_measures = BigramAssocMeasures()

print("Top 10 Bigram Collocations:")
print("\nBy PMI (Pointwise Mutual Information):")
print(bigram_finder.nbest(bigram_measures.pmi, 10))

print("\nBy Chi-Square:")
print(bigram_finder.nbest(bigram_measures.chi_sq, 10))

print("\nBy Likelihood Ratio:")
print(bigram_finder.nbest(bigram_measures.likelihood_ratio, 10))
```

### 11.4 Trigram Collocations

```python
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

# Trigram collocations
trigram_finder = TrigramCollocationFinder.from_words(tokens)
trigram_finder.apply_freq_filter(2)

trigram_measures = TrigramAssocMeasures()

print("Top 10 Trigram Collocations:")
print(trigram_finder.nbest(trigram_measures.pmi, 10))
```

---

## 12. WordNet & Semantic Analysis

### 12.1 WordNet Basics

```python
from nltk.corpus import wordnet as wn

# Get synsets (synonym sets) for a word
synsets = wn.synsets('dog')
print(f"Synsets for 'dog': {synsets}")

# Synset details
dog_synset = wn.synset('dog.n.01')
print(f"\nSynset: {dog_synset.name()}")
print(f"Definition: {dog_synset.definition()}")
print(f"Examples: {dog_synset.examples()}")
print(f"Lemmas: {dog_synset.lemma_names()}")
```

### 12.2 Word Relationships

```python
from nltk.corpus import wordnet as wn

# Synonyms
dog = wn.synset('dog.n.01')
print(f"Synonyms: {dog.lemma_names()}")

# Hypernyms (more general)
print(f"Hypernyms: {dog.hypernyms()}")

# Hyponyms (more specific)
print(f"Hyponyms: {dog.hyponyms()[:5]}")  # First 5

# Holonyms (part of)
print(f"Part holonyms: {dog.part_holonyms()}")
print(f"Member holonyms: {dog.member_holonyms()}")

# Meronyms (has parts)
print(f"Part meronyms: {dog.part_meronyms()}")
print(f"Substance meronyms: {dog.substance_meronyms()}")

# Antonyms (for lemmas, not synsets)
good = wn.synset('good.a.01')
good_lemma = good.lemmas()[0]
print(f"Antonyms of 'good': {good_lemma.antonyms()}")
```

### 12.3 Word Similarity

```python
from nltk.corpus import wordnet as wn

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
car = wn.synset('car.n.01')

# Path similarity (0 to 1)
print(f"dog-cat path similarity: {dog.path_similarity(cat):.3f}")
print(f"dog-car path similarity: {dog.path_similarity(car):.3f}")

# Wu-Palmer similarity (based on depth in taxonomy)
print(f"dog-cat wup similarity: {dog.wup_similarity(cat):.3f}")
print(f"dog-car wup similarity: {dog.wup_similarity(car):.3f}")

# Leacock-Chodorow similarity
print(f"dog-cat lch similarity: {dog.lch_similarity(cat):.3f}")
print(f"dog-car lch similarity: {dog.lch_similarity(car):.3f}")
```

### 12.4 Finding All Synonyms

```python
from nltk.corpus import wordnet as wn

def get_all_synonyms(word):
    """Get all synonyms for a word"""
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def get_all_antonyms(word):
    """Get all antonyms for a word"""
    antonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name().replace('_', ' '))
    return list(antonyms)

print(f"Synonyms of 'happy': {get_all_synonyms('happy')}")
print(f"Antonyms of 'happy': {get_all_antonyms('happy')}")
```

### 12.5 Word Sense Disambiguation

```python
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

# 'bank' has multiple meanings
sentence1 = "I went to the bank to deposit money"
sentence2 = "The river bank was muddy"

# Lesk algorithm
sense1 = lesk(word_tokenize(sentence1), 'bank')
sense2 = lesk(word_tokenize(sentence2), 'bank')

print(f"Sentence 1 - 'bank' sense: {sense1}")
print(f"Definition: {sense1.definition()}")

print(f"\nSentence 2 - 'bank' sense: {sense2}")
print(f"Definition: {sense2.definition()}")
```

---

## 13. Sentiment Analysis

### 13.1 VADER Sentiment Analysis

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Analyze sentiment
texts = [
    "This movie is amazing! I loved every minute of it.",
    "The food was terrible and the service was slow.",
    "It's okay, nothing special but not bad either.",
    "I absolutely HATE this product!!!",
    "The weather is nice today.",
    "That was the BEST experience ever! 😊",
    "This is not good, not good at all."
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"\nText: {text}")
    print(f"Scores: {scores}")
    
    # Interpret compound score
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print(f"Sentiment: {sentiment}")
```

### 13.2 Understanding VADER Scores

```python
"""
VADER returns 4 scores:
- neg: Negative sentiment proportion (0-1)
- neu: Neutral sentiment proportion (0-1)
- pos: Positive sentiment proportion (0-1)
- compound: Normalized compound score (-1 to 1)

Compound score interpretation:
- >= 0.05: Positive
- <= -0.05: Negative
- Between: Neutral

VADER handles:
- Punctuation amplification (Amazing! vs Amazing)
- Capitalization (GREAT vs great)
- Degree modifiers (very good vs good)
- Contrastive conjunctions (but)
- Negations (not good)
- Emojis 😊 😢
"""

sia = SentimentIntensityAnalyzer()

# Demonstrating VADER features
examples = [
    ("Good", "Normal"),
    ("GOOD", "Capitalized"),
    ("Good!", "Punctuation"),
    ("Very good", "Intensifier"),
    ("Good, but not great", "Contrastive"),
    ("Not good", "Negation"),
]

for text, feature in examples:
    score = sia.polarity_scores(text)['compound']
    print(f"{feature:<15} '{text}': {score:.3f}")
```

### 13.3 Sentiment Analysis with Naive Bayes

```python
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import random

# Prepare data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Feature extraction
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Create feature sets
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split data
train_set = featuresets[:1500]
test_set = featuresets[1500:]

# Train classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate
print(f"Accuracy: {accuracy(classifier, test_set):.2%}")

# Most informative features
print("\nMost Informative Features:")
classifier.show_most_informative_features(10)
```

### 13.4 Custom Sentiment Lexicon

```python
# Custom sentiment words
positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful',
                  'fantastic', 'love', 'best', 'happy', 'beautiful'}
negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst',
                  'hate', 'poor', 'disappointing', 'sad', 'ugly'}

def simple_sentiment(text):
    """Simple lexicon-based sentiment analysis"""
    words = set(text.lower().split())
    
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    
    if pos_count > neg_count:
        return 'positive', pos_count - neg_count
    elif neg_count > pos_count:
        return 'negative', neg_count - pos_count
    else:
        return 'neutral', 0

# Test
texts = [
    "This is a great and wonderful product",
    "This is terrible and awful",
    "This is okay"
]

for text in texts:
    sentiment, confidence = simple_sentiment(text)
    print(f"'{text}' -> {sentiment} (confidence: {confidence})")
```

---

## 14. Text Classification

### 14.1 Document Classification Pipeline

```python
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier
from nltk.classify.scikitlearn import SklearnClassifier
import random

# Feature extraction function
def extract_features(words, common_words):
    """Extract features from document words"""
    return {word: (word in words) for word in common_words}

# Prepare data
def prepare_movie_review_data():
    # Get documents
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    
    # Get common words
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words()
                              if w.isalpha())
    common_words = [w for w, _ in all_words.most_common(2000)]
    
    # Extract features
    featuresets = [(extract_features(set(w.lower() for w in doc), common_words), label)
                   for doc, label in documents]
    
    return featuresets, common_words

featuresets, common_words = prepare_movie_review_data()

# Split
train_size = int(0.8 * len(featuresets))
train_set = featuresets[:train_size]
test_set = featuresets[train_size:]

print(f"Training samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")
```

### 14.2 Multiple Classifiers

```python
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.classify.util import accuracy

# Naive Bayes
nb_classifier = NaiveBayesClassifier.train(train_set)
print(f"Naive Bayes Accuracy: {accuracy(nb_classifier, test_set):.2%}")

# Decision Tree (may take longer)
# dt_classifier = DecisionTreeClassifier.train(train_set)
# print(f"Decision Tree Accuracy: {accuracy(dt_classifier, test_set):.2%}")

# Maximum Entropy (logistic regression)
# me_classifier = MaxentClassifier.train(train_set, algorithm='GIS', max_iter=10)
# print(f"MaxEnt Accuracy: {accuracy(me_classifier, test_set):.2%}")
```

### 14.3 Sklearn Integration

```python
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifiers = [
    ('Multinomial NB', MultinomialNB()),
    ('Bernoulli NB', BernoulliNB()),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('SGD Classifier', SGDClassifier(max_iter=1000)),
    ('Linear SVC', LinearSVC(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
]

for name, sklearn_classifier in classifiers:
    classifier = SklearnClassifier(sklearn_classifier)
    classifier.train(train_set)
    acc = accuracy(classifier, test_set)
    print(f"{name}: {acc:.2%}")
```

### 14.4 Cross-Validation

```python
from nltk.classify import accuracy
import numpy as np

def cross_validate(featuresets, classifier_class, n_folds=5):
    """Perform k-fold cross-validation"""
    fold_size = len(featuresets) // n_folds
    accuracies = []
    
    for i in range(n_folds):
        # Split data
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        test_set = featuresets[test_start:test_end]
        train_set = featuresets[:test_start] + featuresets[test_end:]
        
        # Train and evaluate
        classifier = classifier_class.train(train_set)
        acc = accuracy(classifier, test_set)
        accuracies.append(acc)
        print(f"Fold {i+1}: {acc:.2%}")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\nMean Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")
    return mean_acc, std_acc

# Run cross-validation
mean_acc, std_acc = cross_validate(featuresets, NaiveBayesClassifier, n_folds=5)
```

### 14.5 Text Classification with TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import movie_reviews

# Prepare raw text data
positive_texts = [' '.join(movie_reviews.words(fid)) 
                  for fid in movie_reviews.fileids('pos')]
negative_texts = [' '.join(movie_reviews.words(fid)) 
                  for fid in movie_reviews.fileids('neg')]

texts = positive_texts + negative_texts
labels = ['pos'] * len(positive_texts) + ['neg'] * len(negative_texts)

# Shuffle
import random
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# Split
split = int(0.8 * len(texts))
X_train, X_test = texts[:split], texts[split:]
y_train, y_test = labels[:split], labels[split:]

# TF-IDF Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"TF-IDF + Naive Bayes Accuracy: {accuracy:.2%}")

# Predict new text
new_texts = ["This movie was fantastic!", "Terrible waste of time"]
predictions = pipeline.predict(new_texts)
for text, pred in zip(new_texts, predictions):
    print(f"'{text}' -> {pred}")
```

---

## 15. Corpus Management

### 15.1 Built-in Corpora

```python
from nltk.corpus import (
    gutenberg, brown, reuters, movie_reviews, 
    webtext, inaugural, stopwords, wordnet
)

# Gutenberg corpus (classic literature)
print("Gutenberg files:", gutenberg.fileids())
print(gutenberg.raw('austen-emma.txt')[:200])

# Brown corpus (categorized text)
print("\nBrown categories:", brown.categories())
print(brown.words(categories='news')[:20])

# Reuters corpus (news)
print("\nReuters categories:", reuters.categories()[:10])

# Movie reviews corpus
print("\nMovie review categories:", movie_reviews.categories())

# Web text corpus
print("\nWebtext files:", webtext.fileids())

# Inaugural addresses
print("\nInaugural files:", inaugural.fileids()[:5])
```

### 15.2 Corpus Readers

```python
from nltk.corpus import gutenberg

# Different reading methods
# Raw text
raw = gutenberg.raw('austen-emma.txt')
print(f"Characters: {len(raw)}")

# Words
words = gutenberg.words('austen-emma.txt')
print(f"Words: {len(words)}")

# Sentences
sents = gutenberg.sents('austen-emma.txt')
print(f"Sentences: {len(sents)}")

# Paragraphs (if available)
# paras = gutenberg.paras('austen-emma.txt')

# Statistics
for fileid in gutenberg.fileids()[:3]:
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    avg_word_len = num_chars / num_words
    avg_sent_len = num_words / num_sents
    print(f"{fileid}: {avg_word_len:.1f} chars/word, {avg_sent_len:.1f} words/sent")
```

### 15.3 Creating Custom Corpus

```python
from nltk.corpus.reader import PlaintextCorpusReader, CategorizedPlaintextCorpusReader

# Simple plaintext corpus
corpus_root = '/path/to/your/corpus'
# wordlist = PlaintextCorpusReader(corpus_root, '.*\.txt')

# Categorized corpus (folder structure)
# corpus_root = '/path/to/corpus'  # Has subfolders like pos/, neg/
# reader = CategorizedPlaintextCorpusReader(
#     corpus_root,
#     r'(?!\.).*\.txt',
#     cat_pattern=r'(neg|pos)/.*'
# )

# Using the reader
# print(reader.categories())
# print(reader.fileids())
# print(reader.words(categories='pos')[:20])
```

### 15.4 Tagged Corpora

```python
from nltk.corpus import brown, treebank, conll2000

# Brown corpus (tagged)
tagged_words = brown.tagged_words(categories='news')
print(f"Tagged words: {tagged_words[:10]}")

# Treebank corpus
print(f"\nTreebank tagged: {treebank.tagged_words()[:10]}")

# CoNLL 2000 (chunking corpus)
print(f"\nCoNLL chunks: {conll2000.chunked_sents()[0]}")
```

### 15.5 Corpus Statistics

```python
from nltk.corpus import brown
from nltk import FreqDist, ConditionalFreqDist

# Word frequency across genres
cfd = ConditionalFreqDist(
    (genre, word.lower())
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

# Genre vocabulary size
for genre in ['news', 'romance', 'science_fiction'][:3]:
    vocab = set(brown.words(categories=genre))
    print(f"{genre}: {len(vocab)} unique words")

# Lexical diversity
def lexical_diversity(text):
    return len(set(text)) / len(text)

for genre in ['news', 'romance', 'science_fiction']:
    words = brown.words(categories=genre)
    div = lexical_diversity(words)
    print(f"{genre} lexical diversity: {div:.3f}")
```

---

## 16. Advanced Topics

### 16.1 Regular Expression Operations

```python
import re
from nltk.tokenize import regexp_tokenize

text = "Call me at 123-456-7890 or email user@example.com. Visit https://nltk.org"

# Extract patterns
patterns = {
    'phones': r'\d{3}-\d{3}-\d{4}',
    'emails': r'[\w\.-]+@[\w\.-]+',
    'urls': r'https?://[\w\./]+',
    'words': r'\b[a-zA-Z]+\b'
}

for name, pattern in patterns.items():
    matches = re.findall(pattern, text)
    print(f"{name}: {matches}")
```

### 16.2 Information Extraction Pipeline

```python
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

def extract_info(text):
    """Complete information extraction pipeline"""
    
    # 1. Sentence segmentation
    sentences = nltk.sent_tokenize(text)
    
    # 2. Tokenization
    tokenized = [word_tokenize(sent) for sent in sentences]
    
    # 3. POS Tagging
    tagged = [pos_tag(tokens) for tokens in tokenized]
    
    # 4. Named Entity Recognition
    chunked = [ne_chunk(tags) for tags in tagged]
    
    # 5. Relation Extraction (simplified)
    entities = []
    for tree in chunked:
        for subtree in tree:
            if hasattr(subtree, 'label'):
                entity = ' '.join(word for word, tag in subtree)
                entity_type = subtree.label()
                entities.append((entity, entity_type))
    
    return {
        'sentences': sentences,
        'tokens': tokenized,
        'pos_tags': tagged,
        'entities': entities
    }

text = """Apple Inc. CEO Tim Cook announced new products in Cupertino, California. 
The company's stock rose by 5% after the announcement."""

info = extract_info(text)
print("Entities found:", info['entities'])
```

### 16.3 Text Similarity Measures

```python
from nltk.metrics import edit_distance, jaccard_distance, masi_distance
from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity

# String similarity
s1 = "natural language"
s2 = "natural languege"

# Edit distance (Levenshtein)
print(f"Edit distance: {edit_distance(s1, s2)}")

# Jaro similarity
print(f"Jaro similarity: {jaro_similarity(s1, s2):.3f}")

# Jaro-Winkler similarity
print(f"Jaro-Winkler: {jaro_winkler_similarity(s1, s2):.3f}")

# Set-based similarity
set1 = set(s1.split())
set2 = set(s2.split())

print(f"Jaccard distance: {jaccard_distance(set1, set2):.3f}")
```

### 16.4 Sequence Labeling with HMM

```python
from nltk.tag import hmm
from nltk.corpus import treebank

# Prepare data
train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:3500]

# Train HMM tagger
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_data)

# Evaluate
accuracy = hmm_tagger.accuracy(test_data)
print(f"HMM Tagger Accuracy: {accuracy:.2%}")

# Tag new sentence
sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']
tagged = hmm_tagger.tag(sentence)
print(f"Tagged: {tagged}")
```

### 16.5 Grammar and Parsing

```python
from nltk import CFG, PCFG
from nltk.parse import RecursiveDescentParser, ViterbiParser

# Probabilistic CFG
pcfg_grammar = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> DT NN [0.5] | DT JJ NN [0.3] | NP PP [0.2]
    VP -> VB NP [0.6] | VB NP PP [0.4]
    PP -> IN NP [1.0]
    DT -> 'the' [0.6] | 'a' [0.4]
    NN -> 'dog' [0.3] | 'cat' [0.3] | 'park' [0.2] | 'mat' [0.2]
    JJ -> 'big' [0.5] | 'small' [0.5]
    VB -> 'chased' [0.5] | 'saw' [0.5]
    IN -> 'in' [0.5] | 'on' [0.5]
""")

# Viterbi parser for PCFG
viterbi_parser = ViterbiParser(pcfg_grammar)
sentence = ['the', 'dog', 'chased', 'a', 'cat']

for tree in viterbi_parser.parse(sentence):
    print(tree)
    print(f"Probability: {tree.prob()}")
```

### 16.6 Feature-Based Grammar

```python
from nltk import grammar, parse

# Feature-based grammar for agreement
feature_grammar = grammar.FeatureGrammar.fromstring("""
    S -> NP[NUM=?n] VP[NUM=?n]
    NP[NUM=?n] -> DET[NUM=?n] N[NUM=?n]
    VP[NUM=?n] -> V[NUM=?n] NP
    DET[NUM=sg] -> 'this' | 'a'
    DET[NUM=pl] -> 'these' | 'some'
    N[NUM=sg] -> 'dog' | 'cat'
    N[NUM=pl] -> 'dogs' | 'cats'
    V[NUM=sg] -> 'chases' | 'sees'
    V[NUM=pl] -> 'chase' | 'see'
""")

parser = parse.FeatureEarleyChartParser(feature_grammar)

sentences = [
    ['this', 'dog', 'chases', 'a', 'cat'],      # Valid
    ['these', 'dogs', 'chase', 'some', 'cats'], # Valid
    # ['this', 'dogs', 'chase', 'a', 'cat']     # Invalid (agreement error)
]

for sent in sentences:
    trees = list(parser.parse(sent))
    print(f"'{' '.join(sent)}': {len(trees)} parse(s)")
```

---

## 17. Real-World Projects

### 17.1 Spam Classifier

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
import random

class SpamClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.classifier = None
        self.word_features = None
    
    def preprocess(self, text):
        """Clean and tokenize text"""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                  if t.isalpha() and t not in self.stop_words]
        return tokens
    
    def extract_features(self, tokens):
        """Extract features from tokens"""
        return {word: (word in tokens) for word in self.word_features}
    
    def train(self, data):
        """Train classifier on labeled data"""
        # data = [(text, label), ...]
        
        # Get word features
        all_words = []
        processed_data = []
        for text, label in data:
            tokens = self.preprocess(text)
            all_words.extend(tokens)
            processed_data.append((tokens, label))
        
        freq_dist = nltk.FreqDist(all_words)
        self.word_features = [w for w, _ in freq_dist.most_common(1000)]
        
        # Create feature sets
        featuresets = [(self.extract_features(tokens), label) 
                       for tokens, label in processed_data]
        
        # Train
        self.classifier = NaiveBayesClassifier.train(featuresets)
        return self
    
    def predict(self, text):
        """Predict label for new text"""
        tokens = self.preprocess(text)
        features = self.extract_features(tokens)
        return self.classifier.classify(features)
    
    def evaluate(self, test_data):
        """Evaluate on test data"""
        featuresets = [(self.extract_features(self.preprocess(text)), label) 
                       for text, label in test_data]
        return nltk.classify.accuracy(self.classifier, featuresets)

# Example usage
training_data = [
    ("Get rich quick! Buy now!", "spam"),
    ("Meeting scheduled for tomorrow", "ham"),
    ("FREE money waiting for you!!!", "spam"),
    ("Can you review the document?", "ham"),
    ("Congratulations! You've won!", "spam"),
    ("Lunch meeting at noon", "ham"),
]

classifier = SpamClassifier()
classifier.train(training_data)

test_texts = [
    "Win FREE prizes today!",
    "Please send the report",
]

for text in test_texts:
    prediction = classifier.predict(text)
    print(f"'{text}' -> {prediction}")
```

### 17.2 Keyword Extractor

```python
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

class KeywordExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def extract(self, text, top_n=10, include_bigrams=True):
        """Extract keywords from text"""
        # Tokenize and POS tag
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Filter for nouns and adjectives
        keywords = []
        for word, tag in tagged:
            if (word.isalpha() and 
                word not in self.stop_words and
                len(word) > 2 and
                tag.startswith(('NN', 'JJ'))):
                lemma = self.lemmatizer.lemmatize(word)
                keywords.append(lemma)
        
        # Count frequencies
        keyword_freq = Counter(keywords)
        
        # Extract bigrams if requested
        if include_bigrams:
            from nltk import bigrams
            bi_keywords = []
            for i in range(len(tagged) - 1):
                w1, t1 = tagged[i]
                w2, t2 = tagged[i + 1]
                if (t1.startswith('JJ') and t2.startswith('NN')):
                    bigram = f"{w1} {w2}"
                    bi_keywords.append(bigram)
            keyword_freq.update(bi_keywords)
        
        return keyword_freq.most_common(top_n)

# Example
extractor = KeywordExtractor()
text = """
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience. Deep learning is a 
type of machine learning that uses neural networks. Natural language 
processing uses machine learning to understand human language.
"""

keywords = extractor.extract(text, top_n=10)
print("Top Keywords:")
for keyword, count in keywords:
    print(f"  {keyword}: {count}")
```

### 17.3 Text Summarizer

```python
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences"""
        words1 = [w.lower() for w in sent1 if w.isalpha()]
        words2 = [w.lower() for w in sent2 if w.isalpha()]
        
        all_words = list(set(words1 + words2))
        
        vector1 = [1 if w in words1 else 0 for w in all_words]
        vector2 = [1 if w in words2 else 0 for w in all_words]
        
        return 1 - cosine_distance(vector1, vector2)
    
    def build_similarity_matrix(self, sentences):
        """Build similarity matrix for sentences"""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        return similarity_matrix
    
    def summarize(self, text, num_sentences=3):
        """Summarize text using TextRank algorithm"""
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Tokenize words for each sentence
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(tokenized_sentences)
        
        # Create graph and apply PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        
        # Rank sentences
        ranked_sentences = sorted(
            ((scores[i], i, sent) for i, sent in enumerate(sentences)),
            reverse=True
        )
        
        # Get top sentences (maintain original order)
        top_indices = sorted([idx for _, idx, _ in ranked_sentences[:num_sentences]])
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary

# Example
summarizer = TextSummarizer()
text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language, in particular how to program computers to process and analyze large 
amounts of natural language data. The result is a computer capable of understanding 
the contents of documents, including the contextual nuances of the language within them.

NLP combines computational linguistics—rule-based modeling of human language—with 
statistical, machine learning, and deep learning models. Together, these technologies 
enable computers to process human language in the form of text or voice data and to 
understand its full meaning, complete with the speaker or writer's intent and sentiment.

NLP drives computer programs that translate text from one language to another, respond 
to spoken commands, and summarize large volumes of text rapidly—even in real time. 
There's a good chance you've interacted with NLP in the form of voice-operated GPS 
systems, digital assistants, speech-to-text dictation software, customer service 
chatbots, and other consumer conveniences.
"""

summary = summarizer.summarize(text, num_sentences=2)
print("Summary:")
print(summary)
```

### 17.4 Named Entity Tagger

```python
from nltk import pos_tag, ne_chunk, word_tokenize, Tree

class NamedEntityTagger:
    def __init__(self):
        self.entity_types = {
            'PERSON': 'People',
            'ORGANIZATION': 'Organizations',
            'GPE': 'Geo-Political Entities',
            'LOCATION': 'Locations',
            'FACILITY': 'Facilities',
            'DATE': 'Dates',
            'TIME': 'Times',
            'MONEY': 'Money',
            'PERCENT': 'Percentages'
        }
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        sentences = nltk.sent_tokenize(text)
        entities = {etype: [] for etype in self.entity_types}
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            tree = ne_chunk(tagged)
            
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity = ' '.join(word for word, tag in subtree)
                    entity_type = subtree.label()
                    if entity_type in entities:
                        entities[entity_type].append(entity)
        
        # Remove duplicates
        return {k: list(set(v)) for k, v in entities.items() if v}
    
    def tag_text(self, text):
        """Return text with entities marked"""
        entities = self.extract_entities(text)
        tagged_text = text
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                tagged_text = tagged_text.replace(
                    entity, f"[{entity}]({entity_type})"
                )
        
        return tagged_text

# Example
tagger = NamedEntityTagger()
text = """
Apple Inc. announced that CEO Tim Cook will visit the White House in 
Washington D.C. to meet with President Biden. The company, headquartered 
in Cupertino, California, reported $100 billion in revenue.
"""

entities = tagger.extract_entities(text)
print("Extracted Entities:")
for entity_type, entity_list in entities.items():
    if entity_list:
        print(f"  {entity_type}: {entity_list}")

print("\nTagged Text:")
print(tagger.tag_text(text))
```

### 17.5 Question Answering System (Simple)

```python
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleQA:
    def __init__(self, documents):
        """Initialize with a list of documents"""
        self.documents = documents
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        
        # Process and vectorize documents
        self.processed_docs = [self.preprocess(doc) for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(self.processed_docs)
    
    def preprocess(self, text):
        """Preprocess text"""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                  if t.isalpha() and t not in self.stop_words]
        return ' '.join(tokens)
    
    def answer(self, question):
        """Find best matching document for question"""
        # Preprocess question
        processed_q = self.preprocess(question)
        
        # Vectorize question
        q_vector = self.vectorizer.transform([processed_q])
        
        # Calculate similarities
        similarities = cosine_similarity(q_vector, self.doc_vectors)[0]
        
        # Get best match
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score > 0.1:
            return self.documents[best_idx], best_score
        else:
            return "I don't have information about that.", 0

# Example
documents = [
    "NLTK is a leading platform for building Python programs to work with human language data.",
    "Tokenization is the process of breaking text into individual words or sentences.",
    "Stemming reduces words to their root form by removing suffixes.",
    "Lemmatization reduces words to their dictionary form using vocabulary analysis.",
    "Part-of-speech tagging assigns grammatical categories to words.",
    "Named entity recognition identifies and classifies named entities in text."
]

qa = SimpleQA(documents)

questions = [
    "What is NLTK?",
    "How does tokenization work?",
    "What is the difference between stemming and lemmatization?",
    "What is POS tagging?"
]

for question in questions:
    answer, score = qa.answer(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print(f"Confidence: {score:.2f}")
```

---

## Quick Reference

### Common Imports

```python
# Core NLTK
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk

# Stemmers & Lemmatizers
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Corpora & Resources
from nltk.corpus import stopwords, wordnet, brown, gutenberg, movie_reviews

# Classification
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Statistical
from nltk import FreqDist, ConditionalFreqDist, bigrams, trigrams, ngrams

# Collocations
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
```

### Essential Downloads

```python
nltk.download('punkt')              # Tokenizers
nltk.download('stopwords')          # Stop words
nltk.download('wordnet')            # WordNet lexical database
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('maxent_ne_chunker')  # NER chunker
nltk.download('words')              # Word lists
nltk.download('vader_lexicon')      # Sentiment lexicon
nltk.download('brown')              # Brown corpus
nltk.download('movie_reviews')      # Movie reviews corpus
nltk.download('omw-1.4')            # Open Multilingual WordNet
```

---

## Further Resources

- **Official Documentation**: https://www.nltk.org/
- **NLTK Book (Free)**: https://www.nltk.org/book/
- **API Reference**: https://www.nltk.org/api/nltk.html
- **GitHub Repository**: https://github.com/nltk/nltk

---

*This guide covers NLTK version 3.x. Last updated: January 2026*
