# Colab v1 bundle — first GPU push

First Colab adaptation of the pipeline. Script set:
- `train_full_fit.py`: full-fit StrongCNN (100% train, multiple seeds)
- `train_pseudo.py`: pseudo-labeling on high-confidence test predictions
- `make_stack_test.py`: LogisticRegression stacker on CNN + features OOF
- `blend_all.py`: final blending (stack + weighted + raw variants)

Notebook: `MPA_Colab.ipynb`.

**Postmortem**: the stack included in final blends hurt LB (stack OOF=0.9707 but stacked submissions scored 0.968 vs 0.9756 for pure CNN ensemble). See `colab_v2/` for the corrected pipeline.
