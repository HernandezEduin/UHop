# Running PQ and PQL with UHop

The UHop adapter supports the family-controlled, mixed 2-hop/3-hop
[PathQuestion-Family](https://github.com/HalcyonSolutions/PathQuestion-Family)
reconstruction. It keeps the **task-filtered** and **source** graph conditions
separate, as in
[How Benchmark Graph Construction Shapes Multi-Hop KG Navigation](https://arxiv.org/abs/2610.26399).

The original PQ/PQL benchmark data is not redistributed here. Reconstruct it
locally using the published converter.

## 1. Reconstruct the datasets once

In a separate PathQuestion-Family checkout:

```bash
python scripts/reconstruct.py --dataset all --graph both --seed 42 --encode-ids
```

The outputs are under `PathQuestion-Family/generated/`. The reconstruction
seed fixes the family-controlled train/dev/test splits and is **independent**
of the UHop training seed.

## 2. Prepare the UHop datasets

From the **UHop repository root**, with the same Python environment used by UHop:

```bash
pip install -r requirements-theseus.txt

python script/preprocess/prepare_pathquestion.py \
  --dataset pq --graph-regime filtered \
  --reconstruction-root /path/to/PathQuestion-Family/generated

python script/preprocess/prepare_pathquestion.py \
  --dataset pql --graph-regime filtered \
  --reconstruction-root /path/to/PathQuestion-Family/generated
```

For the source-graph intervention, repeat with `--graph-regime source`.
This creates independent directories `data/THESEUS/pq_filtered/`,
`pq_source/`, `pql_filtered/`, and `pql_source/`.

The converter uses the released question-family splits and released answers.
It does not use graph-derived answers as training targets. It supervises UHop's
original relation-choice and termination-decision tasks using the **first
released relation chain** per QA instance by default. When a question has
multiple valid released chains, all of them are retained in evaluation
metadata. To supervise each chain separately instead, pass
`--supervision-chains all`. This expands the examples and can give conflicting
relation-choice labels for the same question. It is an alternative experiment,
not the default protocol.

As with Kinship and MQuAKE-ST, embeddings are 300-dimensional
[NeuML/glove-6B](https://huggingface.co/NeuML/glove-6B) by default. The
`--glove` and `--glove-source local` options allow an existing GloVe text
file. The prepared data is kept outside Git.

## 3. Train and evaluate

From the UHop repository root:

```bash
bash ./configs/pq_train.sh
bash ./configs/pq_eval.sh

bash ./configs/pql_train.sh
bash ./configs/pql_eval.sh
```

These default to the **filtered graph**, seed 42, 100 training epochs, and
automatic device selection. Use `GRAPH_REGIME=source` for the restored graph
in **both** the train and evaluation commands:

```bash
GRAPH_REGIME=source SEED=0 DEVICE=cuda bash ./configs/pq_train.sh
GRAPH_REGIME=source SEED=0 DEVICE=cuda bash ./configs/pq_eval.sh
```

To compare the graph-derived answer reference while retaining the released
reference as the default:

```bash
bash ./configs/pq_eval.sh --answer-reference graph
```

To evaluate all recorded question text variants:

```bash
bash ./configs/pql_eval.sh --question-variant paraphrased
```

The `--split valid` or `--split test` options can be supplied to either
evaluation script. Set `EPOCHS`, `DEVICE`, and `SEED` via environment
variables on training scripts. Optional `PQ_CHECKPOINT_DIR` or
`PQL_CHECKPOINT_DIR` override the default checkpoint location.

## 4. Interpretation of results

- `hits_at_1` is the result of a **gold-blind uniform selection** from UHop's
  relation-derived terminal entity frontier. UHop itself does not rank
  terminal entities.
- `answer_reachability` is an **oracle upper bound**, not Hits@1.
- `RED` is raw ordered relation edit distance. `F1_REL` is relation-type F1.
  If a question has multiple released relation chains, metrics are computed
  against the closest valid reference (minimum RED, maximum F1).
- PQ/PQL summaries include `family_macro` alongside instance-micro `all`
  and `by_hop`. Family-macro first averages questions within each structural
  family, then gives every family equal weight.
- Since UHop scores *relations*, it does not produce an unambiguous sequence
  of entity–relation–entity edges. PED and F1_SG are consequently unavailable.

A checkpoint's `source.json` records the local reconstruction path. If you
move the PathQuestion-Family outputs, supply
`--offline-source /new/path/to/generated` to the evaluation wrapper.
