The scripts and code for lexical induction. 

[fastText](https://github.com/facebookresearch/fastText) is first used to train monolingual word embeddings. 
Then, the [VecMap](https://github.com/artetxem/vecmap) is used to learn the cross-lingual mapping of word embeddings, and finally, the nearest neighbors are computed using [CSLS](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries).



