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
print(output)