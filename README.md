# A Metric Learning-based Method for Biomedical EL

The code for paper [A metric learning-based method for biomedical entity linking](https://www.frontiersin.org/articles/10.3389/frma.2023.1247094/full).

We proposed a simple method based on metric-learning for entity linking task. Our method tackles the imbalance data issue and reduces significantly training and inference cost. We experienced on two challenge datasets which are BC5CDR and MedMention. Experimental results demonstrate that our proposed method achieves competitive linking performance compared to other SOTA methods.


![image](https://github.com/dinhngoc267/a-metric-learning-based-method-for-biomedical-entity-linking/assets/49720223/13d8d191-75ae-4092-997e-05504c31524e)

## Prepare Environment

```
conda create --name el_env python=3.9
conda activate el_env
pip install -r requirements.txt
```

## Dataset 

### MedMention 

ST21pv: https://github.com/chanzuckerberg/MedMentions

You can download the UMLS2017AA here:
https://drive.google.com/file/d/1sbTlpeeoNQQV70JWKVdkSKpf5Xi-EZdE/view?usp=drive_link


## Experiments

```
conda activate el_env
export PYTHONPATH=.
python src/train.py
```


