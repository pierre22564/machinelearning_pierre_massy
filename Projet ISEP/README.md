# Projet ISEP — MPA-MLF Room Occupancy 2026

Kaggle competition: classifier le nombre d'occupants (0, 1, 2, 3) dans une pièce à partir d'images delay-Doppler issues d'un radar mmWave 60 GHz.

- **Data**: 9227 images train + 3955 test (45×51 RGB, paddées à 48×64).
- **Metric**: accuracy.
- **Leaderboard**: 0.97565 (rank #4). Top 1 = 0.97819.

## Structure

```
Projet ISEP/
├── local/          Pipeline initial en local (M-series Mac, MPS)
├── colab_v1/       Première adaptation Colab (CUDA)
├── colab_v2/       Pipeline optimisé (multi-archi, pseudo itératif, no-stack)
└── report.pdf      Rapport final
```

## Pipeline général

1. **Feature engineering** (HGB sur stats d'image + FFT features) — OOF ~0.933
2. **StrongCNN from scratch** (4 blocs conv 3→48→96→192→384 + SiLU + GAP/GMP head) — OOF ~0.968 par seed
3. **Full-fit** sur 100% du train avec multi-seeds + TTA — boost test-time
4. **Pseudo-labeling** sur les prédictions haute-confiance (thr ≥ 0.98) — +0.002 à +0.005
5. **Blending** pondéré (éviter le stacking LogReg — il overfit sur le CV et dégrade le LB)

## Résultats LB

| Submission | LB | Commentaire |
|---|---|---|
| `sub_pseudo_heavy` (local overnight) | **0.97565** | 🥇 best — pure CNN ensemble, no stack |
| `sub_cnn_full_pseudo` | 0.97515 | full-fit + pseudo, no stack |
| `submission_with_fullfit` (Colab v1) | 0.96855 | **stack 50%** → regressed |
| `submission_mega` (Colab v1) | 0.96754 | stack 20% + full + pseudo — stack poison |

**Leçon clé**: le stacker LogReg entraîné sur les OOF CNN + HGB donne un excellent OOF (0.9707) mais dégrade le LB de ~0.007. Raison probable : shift de distribution train/test, ou les CNN OOF sont déjà saturés et le LogReg overfit les patterns d'erreur spécifiques au CV. **Solution v2**: blend purement par moyenne (pondérée) sans stacker.

## Hardware utilisé

- **Local** : MacBook M-series (MPS). Bottleneck: `com.apple.provenance` xattr de macOS cause un hang de `syspolicyd` après ~5000 `open()`. Contourné en pré-chargeant tout en uint8 dans un `.npy` unique.
- **Colab T4** (15 GB VRAM): ~5-10× plus rapide que MPS. Principal outil pour v1 et v2.
