## Hypergraph Clustering based Multi-label Cross-modal Retrieval

We refer to the P-GNN and HGNN implementations to build our code.


## Dependencies

- Python (>=3.8)

- PyTorch (>=1.7.1)

- Scipy (>=1.5.2)

## Datasets
You can download the features of the datasets from:
 - MIRFlickr, [OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn), [BaiduPan(password: b04z)](https://pan.baidu.com/s/1g1c7Ne7y1BDys6pMh2yhYw)
 - NUS-WIDE (top-21 concepts), [BaiduPan(password: tjvo)](https://pan.baidu.com/s/1JEokBLtpQkx8JA1uAhBzxg)
 - MS-COCO, [BaiduPan(password: 5uvp)](https://pan.baidu.com/s/1uoV4K1mBwX7N1TVmNEiPgA)

## Process
 - Place the datasets in `data/`
 - Set the experiment parameters in `main.py`.
 - Train a model:
 ```bash
 python main.py
```
 - Modify the parameter `EVAL = True` in `main.py` for evaluation:
  ```bash
 python main.py
```


