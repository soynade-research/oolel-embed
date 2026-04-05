**!!WORK IN PROGRESS!!** We are still cleaning, uploading and documenting the code
# Oolel-Embed

**Oolel-Embed** is a cross-lingual speech and text embedding model developed for Wolof and French. It enables the direct retrieval of French text documents from Wolof speech queries without relying on intermediate automatic speech recognition and translation pipelines. It utilizes **Matryoshka Representation Learning** to produce embeddings at multiple flexible dimensions, allowing users to balance retrieval performance with computational and storage costs.

This repository provides the training and evaluation pipelines for both text-only and multimodal speech-text representation models. It implements a late-fusion architecture that integrates a **HuBERT** speech encoder with a pre-trained language model, outperforming standard dual-encoder baselines. The trained models support various cross-modal and instruction-following tasks, including: 
- speech-to-document retrieval
- transcription retrieval
- speech intent detection.

# Resources
- [Paper](https://arxiv.org/pdf/2602.19991)

- [Model](https://huggingface.co/soynade-research/Oolel-Embed)

- [Evaluation Datasets](https://huggingface.co/collections/soynade-research/kallaama-retrieval-eval)
