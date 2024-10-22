# XAI Analysis

This folder contains the files for running the explainability analysis on RoBERTa and LLama2 7B models. The `kernelShap.pkl` files contain the attribution scores we computed. The following scripts can be used to compute the attribution score from scratch.

To run the analysis on LLama2 7B you can run the `*.sh` scripts:
```bash
./run_xai_matres.sh
./run_xai_tbdense.sh
./run_xai_timeline.sh
```
Or for RoBERTa model you can run:
```bash
python xai_bert.py --dataset MATRES
python xai_bert.py --dataset TIMELINE
python xai_bert.py --dataset TB-DENSE
```

You can find the code to plot the distributions of the KernelShap scores in `plot_attributes.ipynb`.