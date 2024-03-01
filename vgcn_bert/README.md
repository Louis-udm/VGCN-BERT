---
language: en
tags:
- exbert
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
---

# VGCN-BERT (DistilBERT based, uncased)

This model is a VGCN-BERT model based on [DistilBert-base-uncased](https://huggingface.co/distilbert-base-uncased) version. The original paper is [VGCN-BERT](https://arxiv.org/abs/2004.05707).

> Much progress has been made recently on text classification with methods based on neural networks. In particular, models using attention mechanism such as BERT have shown to have the capability of capturing the contextual information within a sentence or document. However, their ability of capturing the global information about the vocabulary of a language is more limited. This latter is the strength of Graph Convolutional Networks (GCN). In this paper, we propose VGCN-BERT model which combines the capability of BERT with a Vocabulary Graph Convolutional Network (VGCN). Local information and global information interact through different layers of BERT, allowing them to influence mutually and to build together a final representation for classification. In our experiments on several text classification datasets, our approach outperforms BERT and GCN alone, and achieve higher effectiveness than that reported in previous studies.

The original implementation is in my gitlab [vgcn-bert repo](https://github.com/Louis-udm/VGCN-BERT), but recently I updated the algorithm and implemented this new version for integrating in HuggingFace Transformers, the new version has the following improvements:
- Greatly speeds up the calculation speed of embedding vocabulary graph convolutinal network (or Word Graph embedding). Taking CoLa as an example, the new model only increases the training time by 11% compared with the base model
- Updated subgraph selection algorithm.
- Currently using DistilBert as the base model, but it is easy to migrate to other models.
- Provide two graph construction methods in vgcn_bert/modeling_graph.py (the same NPMI statistical method as the paper, and the predefined entity-relationship mapping method)

I hope that after integrating into transformers, someone can discover some more practical use case and share to me. I am ashamed to say that I have not discovered too much real use cases myself, mainly because the word-grounded graph obtained through statistical methods has limited improvement on the LLM model. I think its potential application should be when there are specific/customized graphs that need to be integrated into LLM.

## How to use

```python
import transformers as tfr

# Use DistilBert tokenizer, that is corresponding to the base model of this version
tokenizer = tfr.AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load VGCN-BERT model
model = tfr.AutoModel.from_pretrained(
    "zhibinlu/vgcn-bert-distilbert-base-uncased", trust_remote_code=True,
    # # if you already have WordGraphs (torch sparse) and their id_maps,
    # # you can directly instantiate VGCN-BERT model with your WGraphs (support multiple graphs)
    # wgraphs=wgraph_list, 
    # wgraph_id_to_tokenizer_id_maps=id_map_list
)

# Generator WGraph symmetric adjacency matrix
# 1st method: Build graph using NPMI statistical method from training corpus
# wgraph, wgraph_id_to_tokenizer_id_map = model.wgraph_builder(rows=train_valid_df["text"], tokenizer=tokenizer)
# 2nd method: Build graph from pre-defined entity relationship tuple with weight
entity_relations = [
    ("dog", "labrador", 0.6),
    ("cat", "garfield", 0.7),
    ("city", "montreal", 0.8),
    ("weather", "rain", 0.3),
]
wgraph, wgraph_id_to_tokenizer_id_map = model.wgraph_builder(rows=entity_relations, tokenizer=tokenizer)

# Add WordGraphs to the model
model.set_wgraphs([wgraph], [wgraph_id_to_tokenizer_id_map])

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)
```


## Fine-tune model
It's better fin-tune vgcn-bert model for the specific tasks.

## Citation
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

## License
VGCN-BERT is made available under the Apache 2.0 license.

## Contact
- [Zhibin.Lu](Zhibin.Lu.ai@gmail.com)
- [louis-udm in GitHub](https://github.com/Louis-udm)