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
Total skip-gram pairs computed: 2413084
Filtered skip-gram pairs with target words: 71824
Results written to: newpmi.txt

newpmi.txt
Text file with all collocations of nouns, verbs and adjectives + target words, with respective PMI scores.

gender.py
Takes newpmi.txt (list of all target words + PMI scores for 3 word type collocations) to create lists of merged target word categories and their top-100 highest-PMI adjectives, nouns and verbs.
The target words are merged into a male and female category. Duplicate collocations' PMI scores were averaged.
Results written to: av_gendered_pmi.xlsx

av_gendered_pmi.xlsx
Excel file of merged target word categories and their top-100 highest-PMI adjectives, nouns and verbs.
The target words are merged into a male and female category. Duplicate collocations' PMI scores were averaged.
Also contains a list of the frequency of each individual target word in the sub-corpus. 

pmi_adj.py
Takes newpmi.txt (list of all target words + PMI scores for 3 word type collocations) to create a list of the top-15 highest-PMI collocations of individual target words and adjectives.
Results written to: new_adj.xlsx

pmi_noun.py
Takes newpmi.txt (list of all target words + PMI scores for 3 word type collocations) to create a list of the top-15 highest-PMI collocations of individual target words and nouns.
Results written to: new_noun.xlsx

pmi_verb.py
Takes newpmi.txt (list of all target words + PMI scores for 3 word type collocations) to create a list of the top-15 highest-PMI collocations of individual target words and verbs.
Results written to: new_verb.xlsx


Perplexity Evaluation (PPL)





