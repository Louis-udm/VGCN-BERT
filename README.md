# VGCN-BERT: Augmenting BERT with Graph Embedding for Text Classification

#### Authors: [Zhibin Lu](https://louis-udm.github.io) (zhibin.lu@umontreal.ca), Pan Du (pan.du@umontreal.ca), [Jian-Yun Nie](http://rali.iro.umontreal.ca/nie/jian-yun-nie/) (nie@iro.umontreal.ca)

## Overview
This is the implementation of our ECIR 2020 paper: "VGCN-BERT: Augmenting BERT with Graph Embedding for Text Classification".

If you make use of this code or the VGCN-BERT approach in your work, please cite the following paper:

<<<<<<< HEAD
@inproceedings{DBLP:conf/ecir/LuDN20,
  author    = {Zhibin Lu and
               Pan Du and
               Jian{-}Yun Nie},
  editor    = {Joemon M. Jose and
               Emine Yilmaz and
               Jo{\~{a}}o Magalh{\~{a}}es and
               Pablo Castells and
               Nicola Ferro and
               M{\'{a}}rio J. Silva and
               Fl{\'{a}}vio Martins},
  title     = {{VGCN-BERT:} Augmenting {BERT} with Graph Embedding for Text Classification},
  booktitle = {Advances in Information Retrieval - 42nd European Conference on {IR}
               Research, {ECIR} 2020, Lisbon, Portugal, April 14-17, 2020, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12035},
  pages     = {369--382},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-45439-5\_25},
  doi       = {10.1007/978-3-030-45439-5\_25},
  timestamp = {Tue, 14 Apr 2020 15:30:33 +0200},
  biburl    = {https://dblp.org/rec/conf/ecir/LuDN20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
=======
     @inproceedings{ZhibinluGraphEmbedding,
	     author = {Zhibin Lu and Pan Du and Jian-Yun Nie},
	     title = {VGCN-BERT: Augmenting BERT with Graph Embedding for Text Classification},
	     booktitle = {European Conference on Information Retrieval (ECIR)},
	     year = {2020}
	   }
>>>>>>> 557080eccf4e16b252911017db2e28aca93aeb02

## Requirements
- Python 3.7
- NLTK 3.4
- [PyTorch 1.0](https://pytorch.org)
- [Huggingface transformer 0.6.2](https://github.com/huggingface/transformers/releases/tag/v0.6.2)
- [SST-2](https://github.com/kodenii/BERT-SST2)
- [CoLA](https://github.com/nyu-mll/GLUE-baselines)

## Directory Structure
The *data/* directory contains the original datasets and the corresponding pre-processed data. The *output/* directory saves the model files while training the models. The *pytorch_pretrained_bert/* directory is the module of Huggingface transformer. The other files with the *.py* suffix are models, utility functions, training programs, and prediction programs.

## Running the code

### Pre-processing datasets
Run *prepare_data.py* to pre-process the dataset and generate the vocabulary graph. 

**Examples:**
```
python prepare_data.py --ds cola
```

### Train models

Run *train_vanilla_vgcn_bert.py* to train the Vanilla-VGCN-BERT model. Before training the model, you must ensure that you have pre-processed the dataset.

**Examples:**
```
python train_vanilla_vgcn_bert.py --ds cola
```

Run *train_vgcn_bert.py* to train the VGCN-BERT model. Before training the model, you must ensure that you have pre-processed the dataset.

**Examples:**
```
python train_vgcn_bert.py --ds cola
```
