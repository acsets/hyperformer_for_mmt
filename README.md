# Hyperformer for Multilingual Machine Translation (MMT)
This repo is a modified edition of [Hyperformer](https://github.com/rabeehk/hyperformer) to fit my need for training a low-resource multilingual machine translation model. The example task is ten Spanish to South American Indigenous languages. Dataset is available at Americasnlp 2021 website. However, feel free to modify `tasks.py` for other language pairs.

I added a functionality of dynamic config file generation so that now multiple experiments can be run with different configs simultaneously. In the original program logic, the config file is static. Since the experiments are conducted on a cluster with job scheduler, jobs can be placed in job queue so when and in what order the job will be executed is indefinite. A dynamic config file generation ease the need of human intervention of manually modifying the config file.


# Usage
The code is written for cluster with PBS job scheduler but it can be easily modifed for any other job scheduler by replacing the job submission code snippet at the bottom of /scripts/entrypoint.sh

Paths are empty strings so ensure to assign the paths that suit your need.

After the path assignment and desired modifications, run the training process with

```
bash entrypoint.sh
```



## Bibliography
This repo contains the PyTorch implementation of the ACL, 2021 paper
[Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks](https://aclanthology.org/2021.acl-long.47.pdf).

If you find this work useful, please cite their paper.

```
@inproceedings{karimi2021parameterefficient,
  title={Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks},
  author={Karimi Mahabadi, Rabeeh and Ruder, Sebastian and Dehghani, Mostafa and Henderson, James},
  booktitle={Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```
