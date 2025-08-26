# 📰 Dual-Branch Fake News Detector (BERT + TransE)
## 🌟 Overview
Welcome! 👋 This project is all about detecting fake news using a dual-branch neural framework that fuses text understanding (BERT) with knowledge graphs (TransE).

It’s not just another NLP model — it combines the semantic meaning of words with the real-world facts extracted as triplets (head–relation–tail).
Together, they help the model reason better and say: “FAKE” or “REAL.”

---

## 🧩 How It Works

1. Text Branch (BERT) 📝

    - Encodes the article text into a [CLS] embedding.

2. Knowledge Branch (TransE) 🧠

    - Extracts knowledge triplets (head, relation, tail).

    - Maps them into embeddings using TransE.

    - Aggregates them into a single “knowledge vector.”

3. Fusion Layer 🔗

    - Concatenates [CLS || knowledge].

    - Small MLP classifier → FAKE / REAL.

---

### 📦 Repository Structure
``` bash
Data/                       # datasets (e.g., liar_dataset)
Models/                     # trained models and checkpoints
outputs/                    # logs and results
triplet_extraction/         # REBEL or custom triplet extractor

aggregate_triplets.ipynb    # aggregate extracted triplets
build_transe_dataset.py     # build entity/relation vocabs + dataset
fuse_text_knowledge.ipynb   # fusion experiments
fusion_inference.ipynb      # evaluate fusion model
gui_fusion_demo.py          # Gradio demo app
train_bert_text_branch.ipynb # train text branch (BERT)
train_transe_knowledge_branch.ipynb # train knowledge branch (TransE)
transe_model.py             # TransE + aggregator
triplet_extraction_rebel.ipynb # triplet extraction notebook

requirements.txt
.gitignore
README.md
papers/                      # IEEE template + draft paper
video_demo.mp4              # demo video (optional)
```

---

## 🚀 Getting Started
1️⃣ Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2️⃣ Prepare Data

- Place your dataset (e.g., Liar, FakeNewsNet) in Data/.

- Run aggregate_triplets.ipynb to extract triplets.

- Run build_transe_dataset.py to build vocab files.

3️⃣ Train Models

- train_bert_text_branch.ipynb → fine-tune BERT.

- train_transe_knowledge_branch.ipynb → train TransE.

- fuse_text_knowledge.ipynb → fusion experiments.

4️⃣ Run GUI 🎨
```bash
python gui_fusion_demo.py
```
Launches a Gradio app → type or paste news → get a FAKE/REAL prediction!

---


## 🎥 Demo

(Add a screenshot or embed your video once ready)

---

## 📑 Research Paper

This repo is accompanied by a research paper written in the IEEE Conference Template.
Check the papers/
 folder or view/edit it on Overleaf.

---

## 🛠️ Tech Stack

- Language Models: BERT (HuggingFace Transformers)

- Knowledge Graphs: TransE implementation (PyTorch)

- Fusion: MLP over embeddings

- GUI: Gradio for easy demos

- Training & Experiments: Jupyter notebooks

---

## 📈 Roadmap

- ✅Text branch (BERT)

- ✅Knowledge branch (TransE)

- ✅Fusion experiments

- ✅Gradio demo app

- ✅Add video demo

- ⌛Finalize IEEE research paper

- (Future) Deploy on HuggingFace Spaces 🚀

---

## 🤝 Contributing

Contributions are welcome! 🎉
If you’d like to improve the project, feel free to fork, open issues, or submit PRs.

---

## 📄 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

---

## 🙌 Acknowledgements

- HuggingFace 🤗 for Transformers

- Babelscape REBEL for triplet extraction

- The FakeNewsNet & Liar datasets

- And all open-source contributors who make research possible ❤️

---

## 👨‍🎓 Authors & Supervision

- Author: Mohamed Ahmed Mansour Mahmoud

- Under the supervision of: Professor Ramakrishna


