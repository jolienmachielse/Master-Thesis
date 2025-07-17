# Master-Thesis


train.py 
Training Gensim Word2Vec model on subcorpus
predict.py 
Code to calculate top next-word predictions 



train.py
Trains word2vec model on corpus subset (finalsubset_commoncorpus.json).

finalsubset_commoncorpus.json
Filtered corpus subset.

run.py
Generates PMI scores for collocations of nouns, adjectives and verbs + target words in PoS-tagged sub-corpus(all_pos_tagged.pos_tagged). 
Output file: runpmi.txt

runpmi.txt
Text file with all collocations of nouns, verbs and adjectives + target words, with respective PMI scores.

gender.py
Creates file with merged target word categories and their top-100 highest-PMI adjectives, nouns and verbs.
The target words are merged into a male and female category. Duplicate collocations' PMI scores were averaged.
Output: av_gendered_pmi.xlsx

av_gendered_pmi.xlsx
Excel file of merged target word categories and their top-100 highest-PMI adjectives, nouns and verbs.
The target words are merged into a male and female category. Duplicate collocations' PMI scores were averaged.
Also contains a list of the frequency of each individual target word in the sub-corpus. 
