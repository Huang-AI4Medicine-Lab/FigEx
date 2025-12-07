# FigEx: Aligned Extraction of Scientific Figures and Captions

- [x] Training code for FigEx 
- [ ] Inference pipeline
- [ ] Baseline implementations

---

## Introduction
This is the official code repository for the paper **[FigEx: Aligned Extraction of Scientific Figures and Captions](https://aclanthology.org/2025.findings-emnlp.899/)**. We propose FigEx, a vision-language model to extract aligned pairs of subfigures and subcaptions from scientific papers. FigEx improves subfigure detection AP<sup>b</sup> over Grounding DINO by 0.023 and boosts caption separation BLEU over Llama-2-13B by 0.465.

## Requirement
Install all the required packages.

```
pip install -r requirements.txt
```

## Training Usage (Lenna-based)

We build FigEx on top of the **[Lenna](https://github.com/Meituan-AutoML/Lenna)**, **[LLaVA](https://github.com/haotian-liu/LLaVA)**, and **[YOLOS](https://github.com/hustvl/YOLOS)** pre-trained weights; our implementation is **inspired by** their open-source codebases.


Run the Stage-1 training script for FigEx:

```
bash train_llava.sh
```

Run the Stage-3 training script for FigEx:

```
bash train_yolos_mix.sh
```

## Training Usage (from pre-trained LLaVA)


## Citation
```
@inproceedings{song2025figex,
  title        = {FigEx: Aligned Extraction of Scientific Figures and Captions},
  author       = {Song, Jifeng and Das, Arun and Cui, Ge and Huang, Yufei},
  booktitle    = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  pages        = {16558--16571},
  year         = {2025}
}
```
