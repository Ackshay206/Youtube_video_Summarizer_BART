import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
import re
from nltk.corpus import stopwords
import sklearn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import transformers
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration

link = "https://www.youtube.com/watch?v=kaUajqGp1XE" 
unique_id = link.split("=")[-1]
sub = YouTubeTranscriptApi.get_transcript(unique_id)  
subtitle = " ".join([x['text'] for x in sub])

from nltk.tokenize import sent_tokenize


nltk.download('punkt_tab')

subtitle = subtitle.replace("\n"," ")
sentences = sent_tokenize(subtitle)

organized_sent = {k:v for v,k in enumerate(sentences)}

tf_idf = TfidfVectorizer(min_df=2, 
                                    strip_accents='unicode',
                                    max_features=None,
                                    lowercase = True,
                                    token_pattern=r'w{1,}',
                                    ngram_range=(1, 3), 
                                    use_idf=True,
                                    smooth_idf=True,
                                    sublinear_tf=True,
                                    stop_words = 'english')

sentence_vectors = tf_idf.fit_transform(sentences)
sent_scores = np.array(sentence_vectors.sum(axis=1)).ravel()

N = 3
top_n_sentences = [sentences[index] for index in np.argsort(sent_scores, axis=0)[::-1][:N]]

# mapping the scored sentences with their indexes as in the subtitle
mapped_sentences = [(sentence,organized_sent[sentence]) for sentence in top_n_sentences]
# Ordering the top-n sentences in their original order
mapped_sentences = sorted(mapped_sentences, key = lambda x: x[1])
ordered_sentences = [element[0] for element in mapped_sentences]
# joining the ordered sentence
summary = " ".join(ordered_sentences)

print("\n===== Summarization using TF-IDF =====\n")
print(summary)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

input_tensor = tokenizer.encode( subtitle, return_tensors="pt", max_length=512)

outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)
outputs_tensor
print("\n===== Summarization using BART =====\n")
print(tokenizer.decode(outputs_tensor[0]))


