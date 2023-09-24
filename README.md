# Political Bias Prediction and Explanation Based on Discursive Structure

This repository contains the code for the paper: [An Integrated Approach for Political Bias Prediction and Explanation Based on Discursive Structure](https://aclanthology.org/2023.findings-acl.711/).

## Requirements

- Python 3.9.7
- PyTorch 1.10.0
- NumPy 1.21.2
  
## Data

- Allsides: https://github.com/ramybaly/Article-Bias-Prediction
- Hyperpartisan: https://pan.webis.de/semeval19/semeval19-web/
- POLITICS: https://github.com/launchnlp/POLITICS

### Embeddings

We used pretrained 300D GloVe vector embeddings (https://nlp.stanford.edu/projects/glove/). Put the pretrained embeddings in ```./embeddings/``` folder.
  
## Usage

### Preprocessing

**(!)** All articles must first be segmented into EDUs using the [disCut](https://gitlab.irit.fr/melodi/andiamo/discoursesegmentation/discut22) discourse segmenter.

Articles must then be stored in ```./data/``` folder in JSON format with the following structure for each article:

- '**ID**': unique document identifier.
- '**topic**': article topic if any.
- '**source**': media from which the article originates.
- '**url**': url of the article if any.
- '**source_url**': url of the media if any.
- '**segments**': segmentation of the article into EDUs (in the form of a list of strings).
- '**original_content**': original content of the article in the form of a single string.
- '**authors**': list of authors.
- '**title**': title of the article.
- '**int_label**': political label in an integer value.
- '**text_label**': political label in text format.

### Training

To train the model, run:

```
python train.py --data ./data/allsides/ --embeddings ./embeddings/glove/ --model ./models/my_model/ --epochs 10 --batch-size 8 --lr 0.01 --sem-dim 100 --struct-dim 100 --proj-dim 200 --dann --dann-alpha 0.7 --input-dropout 0.5
```

### Explanations

Explanations are generated using the LIME implementation for texts: [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime).


We rely on the diagnostic properties proposed by [Atanasova et al. (2020)](https://aclanthology.org/2020.emnlp-main.263/) for the evaluation: [https://github.com/copenlu/xai-benchmark](https://github.com/copenlu/xai-benchmark).

# License

This repository is licensed under CC BY-NC-SA 4.0.
