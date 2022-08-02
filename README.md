<img src="src/logo_full_res.png" alt="Logo" width="200"/>

# Z-Fed
## A ZKP Federated Framework to Support Balanced Learning

### Project topics
Development of a data analysis tool for balanced federated machine learning. The proposed framework can be used to classify data more fairly by reducing bias related to the presence of minority groups within the dataset. The project, motivated by recent scientific results, aims to implement unfairness quantification as part of the fundamental principles described in the Ethics guidelines for trustworthy AI authored by the High-Level Expert Group on Artificial Intelligence (AI HLEG) on behalf of the European Commission.
### Research Questions
- To what extent can we mitigate federated learning bias according to EU guidelines for data ethics and trustworthy AI?
- To what extent can zero knowledge proof metadata about the proportions of population groups can be used in a federated learning environment to enhance AI trustworthiness?


## Abstract
Federated learning (FL) is a distributed machine
learning approach that enables remote devices i.e. workers to
collaborate to compute the fitting of a neural network model
without sharing their data. While this method is favorable to 
ensure data privacy, an imbalanced data distribution can introduce
unfairness in the model training, causing discriminatory bias
towards certain under-represented groups. In this paper, we show
that imbalance federated data decreases indexes of statistical
parity difference, equal opportunity difference, and equal odd
difference. To address the problem, we propose a FL framework
called Z-Fed that 1) balances the training without exchange
of privacy protected data using a zero knowledge proof (ZKP)
authentication technique, and 2) allows collecting information on
data distributions based on one or more categorical features to
produce metadata about population proportions. The proposed
framework is able to infer the precise data distribution without
exchanging knowledge of the data and use them to coordinate a
balanced training. Z-Fed aims to mitigate the effect of imbalanced
data in FL without using mediators or probabilistic approaches.
Compared to a non-balanced framework, Z-Fed, on average,
improves the indexes of equal opportunities of 53.54%, the equal
odds of 56.41%, and the statistical parity of 46.1% on imbalanced
UTK datasets, reducing biased predictions among subgroups.
Given the results obtained, Z-Fed can reduce discriminatory
behaviors of FL AI and enhance trustworthy federated learning.

## Paper
[Privacy-enhanced ZKP Framework for Balanced Federated Learning](https://github.com/StefanoMarzo/self-balancing-zkp-federated-learning/blob/main/docs/z-fed.pdf)
