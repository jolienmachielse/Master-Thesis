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
PPL_ratio.py
Calculates perplextity ratio for prompt file with (perplexity for prefixes)/(perplexity for longer prompts)
Results written to results_PPLratio.txt

prompts.adj.txt 
Text file with adjective prompt pairs to calculate PPL on.
10 sets with each 20 sentence containing male-associated adjectives: 10 sentences with a male subject and 10 sentences with a female subject.
10 sets with each 20 sentence containing female-associated adjectives: 10 sentences with a male subject and 10 sentences with a female subject.

prompts.noun.txt
Text file with noun prompt pairs to calculate PPL on.
10 sets with each 20 sentence containing male-associated nouns: 10 sentences with a male subject and 10 sentences with a female subject.
10 sets with each 20 sentence containing female-associated nouns: 10 sentences with a male subject and 10 sentences with a female subject.

prompts.verb.txt
Text file with verb prompt pairs to calculate PPL on.
10 sets with each 20 sentence containing male-associated verbs: 10 sentences with a male subject and 10 sentences with a female subject.
10 sets with each 20 sentence containing female-associated verbs:: 10 sentences with a male subject and 10 sentences with a female subject.
 
New Word Prediction
nwadjectives.py
File contains prompts to elicit NW predictions for nouns. Prediction output: ALL_adj.prediction
Also checks for matches between predictions and the highest-PMI male and female nouns. Match output: nwadjectives.txt
Aggregates predictions for all male prompts and all female prompts, averages probability per predicted word and how many times each individual word is predicted. Output: female_sorted_adj.txt male_sorted_adj.txt

nwnouns.py
File contains prompts to elicit NW predictions for nouns. Prediction output: ALL_noun.prediction
Also checks for matches between predictions and the highest-PMI male and female nouns. Match output: nwnouns.txt
Aggregates predictions for all male prompts and all female prompts, averages probability per predicted word and how many times each individual word is predicted. Output: female_sorted_nouns.txt male_sorted_nouns.txt

nwnverbs.py
File contains prompts to elicit NW predictions for nouns. Prediction output: ALL_verb.prediction
Also checks for matches between predictions and the highest-PMI male and female nouns. Match output: nwverbs.txt
Aggregates predictions for all male prompts and all female prompts, averages probability per predicted word and how many times each individual word is predicted. Output: female_sorted_verbs.txt male_sorted_verbs.txt

