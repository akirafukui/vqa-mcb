# Training

## Models

This folder contains a couple models, described below:

- `v4`: MCB + Genome
- `v4_glove`: MCB + Genome + Glove
- `multi_att_2_glove`: MCB + Genome + Attention + GloVe

## Installing spaCy

Models that use GloVe vectors have additional dependencies. You'll need to install spaCy and download the GloVe vectors:

```
pip install cymem
pip install thinc
pip install https://github.com/spacy-io/spaCy/zipball/master
python -m spacy.en.download
sputnik --name spacy install en_glove_cc_300_1m_vectors
```


## Training

To train a model, edit the corresponding `config.py` and `qlstm_solver.prototxt` files.

In `config.py`, the `TRAIN_DATA_SPLITS`, `QUESTION_VOCAB_SPACE`, and `ANSWER_VOCAB_SPACE` parameters take a `+` delimited string of data sources, which are specified in the `DATA_PATHS` dictionary. We recommend using `train+val+genome` for training.

In `qlstm_solver.prototxt`, set `snapshot` and `snapshot_prefix`  correctly.

Now just run `python train_xxx.py`. Training can take some time. Snapshots are saved according to the settings in `qlstm_solver.prototxt`. To stop training, just hit `Control + C`.