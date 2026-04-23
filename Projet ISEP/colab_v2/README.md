# MPA-MLF v2 — push vers TOP 1

## Diagnostic v1
Le stack LogReg a **tué** le LB : `pseudo_heavy` (no stack) = 0.97565 ✅  
`with_fullfit` (stack 50%) = 0.96855 ❌ (−0.007)

→ v2 vire le stack du final blend.

## Leviers v2

1. **ResNet-CIFAR from scratch** — archi complètement différente de StrongCNN, vraie diversité d'ensemble
2. **StrongCNN base=96** — plus de capacité (T4 à 35% RAM, on a la place)
3. **Pseudo-labeling v2 itératif** — round 1 (thr=0.98) puis round 2 (thr=0.995)
4. **Heavy TTA 5-way** — orig + flip + shifts ±1
5. **batch=512 + num_workers=4** — 2× plus rapide

## Architecture du pipeline

```
Phase 1 (base, no pseudo):
  StrongCNN b=64 × 8 seeds
  StrongCNN b=96 × 5 seeds
  ResNet-CIFAR × 6 seeds

Phase 2: blend_r1 (mean/geom/pow/conf) → pseudo_r1 (thr=0.98)

Phase 3 (pseudo round 1):
  StrongCNN b=64 × 5 seeds (+pseudo)
  StrongCNN b=96 × 3 seeds (+pseudo)
  ResNet × 4 seeds (+pseudo)

Phase 4: blend_r2 → pseudo_r2 (thr=0.995, plus strict)

Phase 5 (pseudo round 2):
  StrongCNN b=64 × 5 seeds (+pseudo v2)
  ResNet × 3 seeds (+pseudo v2)

Phase 6: mega blend (sans stack) = 8 sources mixées
```

## Usage Colab

1. Upload `colab_v2.zip` dans `/content/`
2. Upload `MPA_Colab_v2.ipynb` via File > Upload notebook
3. Runtime > Change runtime type > **T4 GPU** (A100 si Pro)
4. Run toutes les cellules

## Temps estimé

- **T4** : ~90-100 min total
- **A100** : ~35 min (mettre `--batch 768`)

## Submissions à tester (max 3-5 par jour sur Kaggle)

Dans l'ordre :
1. `blend_final/submission_mean.csv` — mean pur des 8 sources
2. `submissions_final/submission_weighted.csv` — poids par round
3. `blend_final/submission_geom.csv` — géométrique (souvent mieux que mean)
4. `submissions_final/submission_arch_balanced.csv` — 50% StrongCNN + 50% ResNet
5. `submissions_final/submission_r3_only.csv` — last round seulement

## Espérance

- LB actuel : 0.97565 (#4, top 1 = 0.97819)
- v2 attendu : 0.978-0.982 → **top 1 sérieusement en vue**
