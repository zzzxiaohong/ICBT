# Iterative Constrained Back-Translation for Unsupervised Domain Adaptation of Machine Translation

Iterative Constrained Back-Translation  (ICBT) apply lexical constraints into back-translation to generate pseudo-parallel data with in-domain lexical knowledge, and then perform round-trip iterations to incorporate more lexical knowledge. We further explore sampling strategies of constrained words in ICBT to introduce more targeted lexical knowledge, via domain specificity and confidence estimation. You can see our COLING 2022 paper "Iterative Constrained Back-Translation for Unsupervised Domain Adaptation of Machine Translation"  for more details.

## Contents

[Downloads](#Downloads)

[Installation](#Installation)

[Usage](#Usage)



[Test](#Test)

## Downloads

We provide the pre-trained model and preprocessed in-domain monolingual data from our experiments, which you can download [here](https://pan.baidu.com/s/1kXxZf19WYnb07WS0JfVy9Q ), and the password is "bb6z". After downloading the data and models, you need to copy the ***din_data*** and ***datamono*** under the ***data*** directory to the ***ICBT/data*** directory, and copy the ***pre-train***, ***language_model*** and ***QE_model*** under the ***models*** directory to the ***ICBT/output*** directory.

***data/din_data***: valid and test data in four domains: *Education*, *Laws*, *Science*, *Thesis*.

***data/datamono***: monolingual data in four domains: *Education*, *Laws*, *Science*, *Thesis*.

***models/pre-train***: neural machine translation models pre-trained on out-of-domain (*News*) data.

***models/language_model***: Chineses masked language models trained on five domains: out-of-domain (*News*), *Education*, *Laws*, *Science*, *Thesis*.

***models/QE_model***: confidence estimation model trained on  out-of-domain (*News*) data.

## Installation

**Install fairseq**

```
cd tools/fairseq && pip install --editable .
```

**Install fastText**

```
cd tools/fastText && mkdir build && cd build && cmake .. && make && make install
```

**Install  transformers**

```
cd tools/transformers && pip install .
cd tools/transformers/examples/pytorch/language-modeling && pip install -r requirements.txt
```



## Usage

### Extract in-domain dictionary

- Train word embeddings on each language:

  ```shell
  bash scripts/lexical_induction/train-embed.sh
  ```

- Build cross-lingual embedding representations:

  ```shell
  bash scripts/lexical_induction/run_map.sh [GPU_ids]
  ```

- Extract dictionary by Cross-domain similarity local scaling ([CSLS](https://arxiv.org/pdf/1710.04087.pdf)):

  ```
  bash scripts/lexical_induction/extract_lexicon.sh
  ```

### Pre-train the models

- Pre-train the en2zh neural machine translation (NMT) model:

  ```
  bash scripts/pre-train/binary-en2zh.sh
  bash scripts/pre-train/train-en2zh.sh [GPU_ids]
  ```

- Pre-train the zh2en constrained back-translation (CBT) model:

  ```
  bash scripts/pre-train/match_replace.sh # constrain the target of out-of-domain parallel data
  bash scripts/pre-train/binary-zh2en.sh 
  bash scripts/pre-train/train-zh2en.sh [GPU_ids]
  ```

- Pre-train the masked language models (MLMs) that will be used in CBT-DomainSpec and ICBT-DomainSpec:

  - Pre-train out-of-domain MLM:

    ```
    bash scripts/MLM-train/train-mlm-dout.sh [GPU_id]
    ```

  - Pre-train in-domain MLM:

    ```
    bash scripts/MLM-train/train-mlm-din.sh [Domain_name] [GPU_id]
    ```

- Train the confidence estimation model that will be used in CBT-Confidence and ICBT-Confidence:

  ```
  bash scripts/QE/train-qe.sh [GPU_id]
  ```

### Fine-tune translation models

- CBT (Lexically constrained back-translation method):

  - Constrain the target in-domain monolingual data:

    - Baseline constraint sampling strategy (CBT-base):

      ```
      bash scripts/CBT-train/match_replace-base.sh [Domain_name]
      ```

    - Or  constraint sampling via domain specificity (CBT-DomainSpec):

      ```
      bash scripts/CBT-train/match_replace-dspec.sh [Domain_name] [GPU_id]
      ```

    - Or  constraint sampling via confidence estimation (CBT-Confidence):

      Firstly, infer the unconstrained data and use the confidence estimation model to score translation quality, and sample the  poorly translated words.

      ```
      bash scripts/CBT-train/inference_for_conf.sh [Domain_name] [GPU_id]
      bash scripts/CBT-train/predict_conf.sh [Domain_name] [GPU_id]
      ```

      And then, match the sampled words with the dictionary and replace them.

      ```
      bash scripts/CBT-train/match_replace-conf.sh [Domain_name] [GPU_id]
      ```

  - Infer constrained monolingual data using pre-trained CBT model:

    ```
    bash scripts/CBT-train/pre-inference-zh2en.sh [Domain_name] [GPU_id] [Sampling_way]
    ```

    For the "Sampling_way" parameter, you can choose *base/dspec/conf*, as does this parameter that appears below.

  - Create the pseudo-parallel data:

    ```
    bash scripts/CBT-train/create_pse_data.sh [Domain_name] [Sampling_way]
    ```

  - Binarize pseudo-parallel data:

    ```
    bash scripts/CBT-train/binary_finetune.sh [Domain_name] [Sampling_way]
    ```

  - Fine-tune the NMT model:

    ```
    bash scripts/CBT-train/fine-tune.sh [Domain_name] [GPU_id] [Sampling_way]
    ```

- ICBT-Base (Baseline iterative constrained back-translation method):

  

- ICBT-DomainSpec(Iterative constrained back-translation method + domain specificity sampling strategy)

  

- ICBT-Confidence(Iterative constrained back-translation method + confidence estimation sampling strategy)

### Test

