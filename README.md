# QK

This codebase contains all the necessary files to conduct all experiments presented in our paper.

# Toy Model:

To train the toy model:

```
cd src
PYTHONPATH=. python toy/recall.py --N 3 --M 3 --d-head 16
PYTHONPATH=. python toy/recall.py --N 5 --M 3 --d-head 16
PYTHONPATH=. python toy/recall.py --N 7 --M 3 --d-head 16
PYTHONPATH=. python toy/recall.py --N 3 --M 5 --d-head 16
PYTHONPATH=. python toy/recall.py --N 5 --M 5 --d-head 16
PYTHONPATH=. python toy/recall.py --N 7 --M 5 --d-head 16
PYTHONPATH=. python toy/recall.py --N 3 --M 7 --d-head 16
PYTHONPATH=. python toy/recall.py --N 5 --M 7 --d-head 16
PYTHONPATH=. python toy/recall.py --N 7 --M 7 --d-head 16
```

Optionally run these with `--gaussian-dgp` to run task variant 2.

These will save checkpoints to `src/toy_model_checkpoints/dhead_16` or `src/toy_model_checkpoints/dhead_16_gaussian.



To extract ranks (Figure 2):
```
PYTHONPATH=. python toy/plot_ranks.py --out-dirs toy_model_checkpoints/dhead_16 --save-dir [save path]
```

To run PCA (Figure 3):
```
PYTHONPATH=. python toy/plot_3d_pca.py --run-dir toy_model_checkpoints/dhead_16/N3_M5_dh16_seed11 --connect_key_menas_hypercube --save-path [save path]
```

To run causal interventions (Figure 4):
```
PYTHONPATH=. python toy/causal_interv.py
```

To plot latent feature interactions (Figure 5):
```
PYTHONPATH=. python toy/plot_G.py
```

# Filter Heads

PCA (Figure 6):
```
cd src
PYTHONPATH=. python categories/pca.py
```
This will save to figures/category_subspace_pca/L20_H26

Causal interventions (Figre 7)
```
cd src
PYTHONPATH=. python categories/causal_interv.py
PYTHONPATH=. python categories/plot_interv_results.py --results ../figures/category_causal_intervention/category_causal_intervention_results.pkl --out [output path]
```

# Binding

PCA (Figure 8):
```
cd src
PYTHONPATH=. python binding/pcas_and_umap.py
```

Causal interventions (Figure 9):
```
cd src 
PYTHONPATH=. python binding/causal_interv.py
PYTHONPATH=. python binding/plot_causal_interv.py
```

Logit Attributions (Figure 10):
```
cd src
PYTHONPATH=. python binding/logits_explained.py
```
