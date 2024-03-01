# VGCN-BERT: Augmenting BERT with Graph Embedding for Text Classification

#### Authors: [Zhibin Lu](https://louis-udm.github.io) (zhibin.lu@umontreal.ca), Pan Du (pan.du@umontreal.ca), [Jian-Yun Nie](http://rali.iro.umontreal.ca/nie/jian-yun-nie/) (nie@iro.umontreal.ca)

## Overview
This is the implementation of our ECIR 2020 paper: "VGCN-BERT: Augmenting BERT with Graph Embedding for Text Classification".

If you make use of this code or the VGCN-BERT approach in your work, please cite the following paper:

     @inproceedings{ZhibinluGraphEmbedding,
	     author    = {Zhibin Lu and Pan Du and Jian-Yun Nie},
	     title     = {VGCN-BERT: Augmenting BERT with Graph Embedding for Text Classification},
	     booktitle = {Advances in Information Retrieval - 42nd European Conference on {IR}
                           Research, {ECIR} 2020, Lisbon, Portugal, April 14-17, 2020, Proceedings,
                           Part {I}},
  	     series    = {Lecture Notes in Computer Science},
  	     volume    = {12035},
  	     pages     = {369--382},
  	     publisher = {Springer},
  	     year      = {2020},
	   }

## ***** New VGCN-BERT model updated in July, 2023 *****
[New Version README](./vgcn_bert/README.md)

The newest version is also in HuggingFace model hub: [vgcn-bert-distilbert-base-uncased](https://huggingface.co/zhibinlu/vgcn-bert-distilbert-base-uncased). The new version has the following improvements:
- Greatly speeds up the calculation speed of embedding vocabulary graph convolutinal network (or Word Graph embedding). Taking CoLa as an example, the new model only increases the training time by 11% compared with the base model
- Updated subgraph selection algorithm.
- Currently using DistilBert as the base model, but it is easy to migrate to other models.
- Provide two graph construction methods in vgcn_bert/modeling_graph.py (the same NPMI statistical method as the paper, and the predefined entity-relationship mapping method)

## Old version
[Old version README](./old_version/README.md)
