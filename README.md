# FigEx: Aligned Extraction of Scientific Figures and Captions

- [x] Training code for FigEx 
- [ ] Inference pipeline (Lenna-based)
- [ ] Baseline implementations

---

## Training Usage (Lenna-based)

We build FigEx on top of the **[Lenna](https://github.com/Meituan-AutoML/Lenna)** pre-trained weights.

Run the Stage-2 training script for FigEx:

```bash
bash train_llava.sh

Run the Stage-3 training script for FigEx:

```bash
bash train_yolos_mix.sh

## Training Usage (from pre-trained LLaVA)

@inproceedings{song2025figex,
  title        = {FigEx: Aligned Extraction of Scientific Figures and Captions},
  author       = {Song, Jifeng and Das, Arun and Cui, Ge and Huang, Yufei},
  booktitle    = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  pages        = {16558--16571},
  year         = {2025}
}
