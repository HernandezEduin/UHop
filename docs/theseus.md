# THESEUS datasets for UHop

This adapter runs the original UHop **relation-choice (RC)** and **termination-decision (TD)** training on the navigation-ready [Kinship](https://huggingface.co/datasets/HalcyonSolutions/Kinship) and [MQuAKE-ST](https://huggingface.co/datasets/HalcyonSolutions/MQuAKE-ST) releases from [Theseus in the Graph](https://arxiv.org/abs/2609.14528).

It preserves the original model architecture, pairwise margin-ranking loss, optimizer choices, stop/continue comparisons, negative sampling and early stopping. Preprocessing provides relation candidates from the released directed KG. UHop predicts a **relation sequence**, not a unique entity-edge trajectory.

## Setup

Use an environment with Python 3.9+ and a compatible PyTorch installation. The original requirements.txt contains historical pinned dependencies and is not required for this dataset adapter.

From the repository root:

~~~bash
pip install -r requirements-theseus.txt
mkdir -p saved_model
~~~

The original UHop dataset download host is suspended, so the original UHop_data.zip is not a prerequisite for these datasets.

The adapter uses the 300-dimensional [NeuML/glove-6B](https://huggingface.co/NeuML/glove-6B) vectors from Hugging Face by default. It downloads only model.safetensors and vocab.json, selects words used by the released QA and relation labels, and writes compact UHop embedding files. Pretrained word vectors remain frozen, as in the original UHop implementation. It does not download the separate SQLite version or export an intermediate full-text GloVe file.

## Prepare

~~~bash
python script/preprocess/prepare_theseus.py --dataset kinship

python script/preprocess/prepare_theseus.py --dataset mquake_st_single

python script/preprocess/prepare_theseus.py --dataset mquake_st_multi
~~~

The data and embedding downloads use the standard Hugging Face cache, so the same 300d GloVe model can be reused across datasets. To make the source explicit, the default is equivalent to:

~~~bash
python script/preprocess/prepare_theseus.py \
  --dataset kinship \
  --glove-source huggingface \
  --glove-model NeuML/glove-6B
~~~

For offline use or another pretrained GloVe text file, retain the original local option:

~~~bash
python script/preprocess/prepare_theseus.py \
  --dataset kinship \
  --glove /path/to/glove.6B.300d.txt
~~~

Supplying --glove automatically selects the local file. The Hugging Face model has only 300-dimensional vectors. To use a local 50-dimensional GloVe file, pass --emb-size 50 as well.

The script fetches the corresponding KG, metadata and QA files from Hugging Face. Alternatively use --offline-source /path/to/downloaded/dataset. For reproducibility, pin --revision to the dataset commit and --glove-revision to the embedding-model commit independently. The chosen embedding source is recorded in source.json. With --output-root, pass the matching prepared directory using --data_dir when training.

Output lives in the ignored data/THESEUS/DATASET/ directory:

- train_data.txt, valid_data.txt, test_data.txt: original UHop relation candidate step structure
- rela2id.json, relation_text.json, word2id.json, word_emb.npy: encoder inputs
- SPLIT_metadata.jsonl: question IDs, sources, answers, reference paths and gold relations
- source.json: dataset/protocol provenance

The official SplitLabel assignments are retained. Only the kg/orig/triplets.txt Kinship graph is used, not its gender-augmented graph. MQuAKE-ST uses kg/triplets.txt without reversing edges. The single-answer and multi-answer settings are separate and should **not** be mixed without a unified split.

For each gold relation chain, the converter aggregates outgoing relations over the KG entities reachable from the stated source by that chain's prefix. The positive relation is marked at each hop. The final step contains outgoing relations marked negative for UHop's stop decision. All supplied answers and reference paths remain in sidecar metadata. A graph mismatch fails preprocessing rather than silently adding gold candidates.

## Train and evaluate

Run the existing UHop entry point from src/, as in the original project:

~~~bash
cd src
python main.py --train --framework UHop --model HR_BiLSTM_plus \
  --dataset kinship --emb_size 300 --hidden_size 100 \
  --dynamic recurrent --epoch_num 100 --device auto
~~~

Change --dataset to mquake_st_single or mquake_st_multi. ABWIM_plus is also available. All original loss weights, optimizer options and training switches remain supported. --num_workers defaults to 0 for portability. Use --device cuda to require GPU.

The existing --test path computes **teacher-forced RC/TD accuracy** against the gold candidate sequences:

~~~bash
python main.py --test --framework UHop --model HR_BiLSTM_plus \
  --dataset kinship --emb_size 300 --hidden_size 100 \
  --dynamic recurrent --path ../saved_model/HR_BiLSTM_plus_0
cd ..
~~~

Use the exact saved hyperparameters from args.txt when testing, not necessarily the illustration above.

For free-running evaluation, use the separate inference-only wrapper:

~~~bash
python script/evaluate_theseus.py \
  --checkpoint saved_model/HR_BiLSTM_plus_0 --split test
~~~

It uses the original trained scoring model and the same RC/TD comparisons to choose a variable-length relation chain. It reports:

- Relation-chain exact-match accuracy
- Answer reachability: at least one gold entity in the terminal relation-derived frontier
- Answer-set exact match and mean terminal frontier size

**These metrics are not entity-level Hits@1, PED or F1_SG.** One selected relation can traverse to many KG entities, and the original UHop policy does not disambiguate among their outgoing edges. The wrapper preserves those possibilities as a set. An entity-level navigation baseline would require an explicit entity-edge policy or a separately specified entity disambiguator, which would change the experimental method. Do not compare relation-frontier reachability directly against THESEUS entity-navigation Hits@1.

The original UHop paper is [Chen et al., NAACL 2019](https://aclanthology.org/N19-1031/). Please also cite the [THESEUS paper](https://arxiv.org/abs/2609.14528) and the dataset upstream sources.
