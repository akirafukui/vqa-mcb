# Multimodal Compact Bilinear Pooling for VQA

This is the code that we wrote to train the state-of-the-art VQA models [described in our paper](https://arxiv.org/abs/1606.01847). Our ensemble of 7 models obtained **66.67%** on real open-ended test-dev and **70.24%** on real multiple-choice test-dev.

## Live Demo

You can upload your own images and ask the model your own questions. [Try the live demo!](http://demo.berkeleyvision.org/)

## Pretrained Model

We are releasing the “MCB + Genome + Att. + GloVe” model from the paper, which achieves **65.38%** on real open-ended test-dev. This is our best individual model.

[Download](https://www.dropbox.com/s/o19k39lvt5cm0bc/multi_att_2_glove_pretrained.zip?dl=0)

You can easily use this model with our evaluation code or with our demo server code.

## Prerequisites

In order to use our pretrained model:

- Compile the `feature/20160617_cb_softattention` branch of [our fork of Caffe](https://github.com/akirafukui/caffe/). This branch contains Yang Gao’s Compact Bilinear layers ([dedicated repo](https://github.com/gy20073/compact_bilinear_pooling), [paper](https://arxiv.org/abs/1511.06062)) released under the [BDD license](https://github.com/gy20073/compact_bilinear_pooling/blob/master/caffe-20160312/LICENSE_BDD), and Ronghang Hu’s Soft Attention layers ([paper](https://arxiv.org/abs/1511.03745)) released under BSD 2-clause.
- Download the [pre-trained ResNet-152 model](https://github.com/KaimingHe/deep-residual-networks).

If you want to train from scratch, do the above plus:

- Download the [VQA tools](https://github.com/VT-vision-lab/VQA).
- Download the [VQA real-image dataset](http://visualqa.org/download.html).
- Optional: Install spaCy and download GloVe vectors. The latest stable release of spaCy has a bug that prevents GloVe vectors from working, so you need to install the HEAD version. See `train/README.md`.
- Optional: Download [Visual Genome](https://visualgenome.org/) data.

## Data Preprocessing

See `preprocess/README.md`.

## Training

See `train/README.md`.

## Evaluation

To generate an answers JSON file in the format expected by the VQA evaluation code and VQA test server, you can use `eval/ensemble.py`. This code can also ensemble multiple models. Running `python ensemble.py` will print out a help message telling you what arguments to use.

## Demo Server

The code that powers our [live demo](http://demo.berkeleyvision.org/) is in `server/`. To run this, you’ll need to install Flask and change the constants at the top of `server.py`. Then, just do `python server.py`, and the server will bind to `0.0.0.0:5000`.

## License and Citation

This code and the pretrained model is released under the BSD 2-Clause license. See `LICENSE` for more information.

Please cite [our paper](https://arxiv.org/abs/1606.01847) if it helps your research:

```
@article{fukui16mcb,
  title={Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding},
  author={Fukui, Akira and Park, Dong Huk and Yang, Daylen and Rohrbach, Anna and Darrell, Trevor and Rohrbach, Marcus},
  journal={arXiv:1606.01847},
  year={2016},
}
```
