# Master-Thesis

## 📁 Project Structure

Master-Thesis/
├── data/
│   └── finalsubset_commoncorpus.json
├── models/
│   └── word2vec.model
├── outputs/
│   ├── NW/
│   │   ├── ALL_adj.predictions.txt
│   │   ├── female_sorted_verbs.txt
│   │   └── ...
│   ├── Collocation/
│   │   ├── av_gendered_pmi.xlsx
│   │   ├── new_adj.xlsx
│   │   └── ...
│   └── PPL/
│       ├── results_PPLratio.txt
│       └── significance_results.txt
├── scripts/
│   ├── CollocationAnalysis/    # PMI scripts
│   ├── NW/                    # New word prediction scripts
│   └── PPL/                   # Perplexity evaluation scripts
├── README.md

##  How to Run

1. **POS Tagging**
python pos_tag_parallel.py --input ./test_pos --ext ".json" --model en_core_web_md --workers 4


2. **PMI Calculation**
cd scripts/CollocationAnalysis
python run.py

3. **Train Word2Vec**
python train.py

4. **Similarity Calculation**
python similarity.py

## Output Files

- `newpmi.txt`: All collocations and PMI scores
- `av_gendered_pmi.xlsx`: Gender-merged PMI lists
- `female_sorted_adj.txt`, `male_sorted_adj.txt`: Sorted NW predictions
- `PMI_similarity_results.txt`: Cosine similarity scores
