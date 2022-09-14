# Iterative Constrained Back-Translation for Unsupervised Domain Adaptation of Machine Translation

Iterative Constrained Back-Translation  (ICBT) apply lexical constraints into back-translation to generate pseudo-parallel data with in-domain lexical knowledge, and then perform round-trip iterations to incorporate more lexical knowledge. We further explore sampling strategies of constrained words in ICBT to introduce more targeted lexical knowledge, via domain specificity and confidence estimation. You can see our COLING 2022 paper "Iterative Constrained Back-Translation for Unsupervised Domain Adaptation of Machine Translation"  for more details.

## Downloads

We provide the pre-trained model and preprocessed in-domain monolingual data from our experiments, which you can download [here](https://pan.baidu.com/s/1kXxZf19WYnb07WS0JfVy9Q ), and the password is "bb6z". After downloading the data and models, you need to copy the ***din_data*** and ***datamono*** under the ***data*** directory to the ***ICBT/data*** directory, and copy the ***pre-train***, ***language_model*** and ***QE_model*** under the ***models*** directory to the ***ICBT/output*** directory.

***data/din_data***: valid and test data in four domains: *Education*, *Laws*, *Science*, *Thesis*.

***data/datamono***: monolingual data in four domains: *Education*, *Laws*, *Science*, *Thesis*.

***models/pre-train***: nmt models pre-trained on out-of-domain (*News*) data.

***models/language_model***: Chineses masked language models trained on five domains: out-of-domain (*News*), *Education*, *Laws*, *Science*, *Thesis*.

***models/QE_model***: confidence estimation model trained on  out-of-domain (*News*) data.

# Installation

##### Install fairseq

```
$ cd tools/fairseq && pip install --editable .
```

##### Install fastText

```
$ cd tools/fastText && mkdir build && cd build && cmake .. && make && make install
```

##### Install  transformers

```
$ cd tools/transformers && pip install .
$ cd tools/transformers/examples/pytorch/language-modeling && pip install -r requirements.txt
```



## Get Started

#### Extract an in-domain dictionary using lexical induction

- Train word embeddings on each language:

  ```shell
  $ bash scripts/lexical_induction/train-embed.sh
  ```

- Build cross-lingual embedding representations:

  ```shell
  $ bash scripts/lexical_induction/run_map.sh  GPU_ids
  ```

- Extract dictionary by Cross-domain similarity local scaling ([CSLS](https://arxiv.org/pdf/1710.04087.pdf)):

  ```
  $ bash scripts/lexical_induction/extract_lexicon.sh
  ```

#### Pre-train the models with out-of-domain data

- Pre-train the en2zh NMT model:

  ```
  $ bash scripts/pre-train/binary-en2zh.sh
  $ bash scripts/pre-train/train-en2zh.sh
  ```

- 
