# Few-shot weld defect images classification using graph neural network
This project finished by Rongdi Wang, Zhenhao He, Jianchao Zhu, Zhisong Zhang, Haiqiang Zuo.

## Abstract
Although the application of deep learning has led to tremendous progress in defect detection technology in recent years, there are still many limitations and bottlenecks. This approach usually requires a large number of high-quality labeled samples, which require skilled workers to collect and label them, but in fact, there are many scenarios in which such requirements are not met. Therefore, in this paper, we design a graph neural network-based algorithm for the detection of weld defects in industrial environments, especially for weld seams, which requires only a very small number of labeled samples to achieve high performance. In addition, we propose a patch mechanism to reduce the computational effort and combine it with a self-attention mechanism to improve the expressiveness of the model. On the public datasets al5083 and ss304, our proposed model achieves 82.0\% and 94.8\% accuracy with only 5\% of the training set, which is 5.2\% and 2.4\% higher than the classical CNN network ResNet18, respectively. The experimental results show that our model has good generalization performance, effectively reduces the consumption of collecting industrial data samples, and also provides the possibility of accurate detection for some special business scenarios.

## Contribution
This article makes the following contributions:

1. Design a neural network with strong generalization performance for a small sample environment.

2. Since graphs usually consist of a large number of nodes and edges and consume a lot of arithmetic power for graph operations, we optimize the graph construction to reduce the amount of operations.

3. To enhance the expressiveness of different graph nodes, we combine self-attention in such a way that the features of different nodes are scaled.

## User's Guide
First install the relevant packages.

`pip install -r requirements.txt`

Second modify the 'ann_path' variable in /utils/resplit_data.py and set it to the location of your '.json' file.

Then run /utils/resplit_data.py generate a '*-few-shot-train.json' file containing only 5% of the training set

`python utils/resplit_data.py`

Last run 'train.py' generate results.

`python train.py`