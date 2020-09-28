# Transformer for English-to-Spanish Translation

Transformer on Tensorflow 2.0 based on paper "Attention Is All You Need". This transformer is for english-to-spanish translation. 
The dataset used is Europarl which can be found here:

https://www.statmt.org/europarl/

Paper: Europarl: A Parallel Corpus for Statistical Machine Translation, Philipp Koehn, MT Summit 2005

The train set has about 1,965,734 english spanish pairs. After removing sentences longer than 40 tokens, I ended up with

* train   ~1,300,000
* test    ~  200,000


Parameters: 
* batch size    256
* dff           1024
* d_model       256
* heads         8

It trained for about 12 hours. Each batch of 256 pairs took about ~22 seconds

Examples of predicted spanish sentences: link

#### Quick Start
To get started do the following:

1. Download Data: http://www.rcalix.com/research/transformers/spanish/
2. Run the code in Tensorflow 2.0

#### Additional Information
Video: 

### Hardware

* RTX 2080 ti 11 vram GPU
* Python: 3.7 or greater
* Linux

### Contact
Ricardo A. Calix, Ph.D.

### Notices
Released as is with no warranty
