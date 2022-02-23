Demo code for "[Causal Discovery in Linear Structural Causal Models with Deterministic Relations](https://openreview.net/forum?id=cU5EeCQk5LX)" by Yuqin Yang, Mohamed Nafea, AmirEmad Ghassami and Negar Kiyavash, CLeaR 2022. 

### Requirements
To install requirements: 
```
pip install -r requirements.txt
```

### Satisfiability

For testing the satisfiability of the derived conditions (Appendix G.1), run:
```
python satisfiability.py
```
The script will return the number of attempts to obtain a linear P-SCM that satisfies Conditions 1 and 2 for given parameters, as well as the matrices A, B of this linear P-SCM.

### Recovery

For testing the performance of P-SCM Recovery algorithm (Appendix G.2), run:
```
python recovery.py
```
Here we use a linear DS-P-SCM without latent confounders (i.e., B is identity) as the generating model, and show how the P-SCM Recovery algorithm works given observed data.
