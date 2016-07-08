# Data preprocessing

## Extracting image features

Edit config settings in `config.py`, then run `python extract_resnet.py`.

Tip: If you have multiple GPUs, you can parallelize the feature extraction by extracting `train2014`, `val2014`, and `test2015` on separate GPUs. You’ll need to change the code in `__main__` in `extract_resnet.py`.

## Preparing the Visual Genome QA pairs

Clone the [Visual Genome Python Driver](https://github.com/ranjaykrishna/visual_genome_python_driver) repo into this directory. Download images, image meta data, and question answers from [Visual Genome website](https://visualgenome.org/api/v0/). The image meta data json file and question answers json file should be in a directory named genome. Then use `vg_preprocessing.py` to convert the Visual Genome QA file(s) into the format we need. 
