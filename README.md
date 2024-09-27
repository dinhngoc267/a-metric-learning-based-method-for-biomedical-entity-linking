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

## Citation 

```
@ARTICLE{10.3389/frma.2023.1247094,

AUTHOR={Le, Ngoc D.  and Nguyen, Nhung T. H. },

TITLE={A metric learning-based method for biomedical entity linking},

JOURNAL={Frontiers in Research Metrics and Analytics},

VOLUME={8},

YEAR={2023},

URL={https://www.frontiersin.org/journals/research-metrics-and-analytics/articles/10.3389/frma.2023.1247094},

DOI={10.3389/frma.2023.1247094},

ISSN={2504-0537},

ABSTRACT={<p>Biomedical entity linking task is the task of mapping mention(s) that occur in a particular textual context to a unique concept or <italic>entity</italic> in a knowledge base, e.g., the Unified Medical Language System (UMLS). One of the most challenging aspects of the entity linking task is the ambiguity of mentions, i.e., (1) mentions whose surface forms are very similar, but which map to different entities in different contexts, and (2) entities that can be expressed using diverse types of mentions. Recent studies have used BERT-based encoders to encode mentions and entities into distinguishable representations such that their similarity can be measured using distance metrics. However, most real-world biomedical datasets suffer from severe imbalance, i.e., some classes have many instances while others appear only once or are completely absent from the training data. A common way to address this issue is to down-sample the dataset, i.e., to reduce the number instances of the majority classes to make the dataset more balanced. In the context of entity linking, down-sampling reduces the ability of the model to comprehensively learn the representations of mentions in different contexts, which is very important. To tackle this issue, we propose a metric-based learning method that treats a given entity and its mentions as a whole, regardless of the number of mentions in the training set. Specifically, our method uses a triplet loss-based function in conjunction with a clustering technique to learn the representation of mentions and entities. Through evaluations on two challenging biomedical datasets, i.e., MedMentions and BC5CDR, we show that our proposed method is able to address the issue of imbalanced data and to perform competitively with other state-of-the-art models. Moreover, our method significantly reduces computational cost in both training and inference steps. Our source code is publicly available <ext-link ext-link-type="uri" xlink:href="https://github.com/dinhngoc267/A-Metric-Learning-based-Method-for-Biomedical-Entity-Linking.git" xmlns:xlink="http://www.w3.org/1999/xlink">here</ext-link>.</p>}}

```
