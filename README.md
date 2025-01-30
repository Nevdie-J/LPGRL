# Self-Supervised Graph Representation Learning via Exploring Complex Linking Semantics

# Requirements

+ CUDA 11.1
+ torch==1.9.1+cu111
+ torch_geometric==2.1.0
+ torch-cluster==1.5.9
+ torch-scatter==2.0.7
+ torch-sparse==0.6.10

# Installation

```bash
pip install -r requirements.txt
```

# Running
## Performance on Link Prediction Task
```bash
. script/lpgrl/run_cora.sh -task link_predictiion
. script/lpgrl/run_cornell.sh -task link_predictiion
. script/lpgrl/run_twitch-de.sh -task link_predictiion
. script/lpgrl/run_twitch-en.sh -task link_predictiion
```

## Performance on Node Classification Task
```bash
. script/lpgrl/run_cora.sh -task node_classification
. script/lpgrl/run_cornell.sh -task node_classification
. script/lpgrl/run_actor.sh -task node_classification
. script/lpgrl/run_blogcatalog.sh -task node_classification
```