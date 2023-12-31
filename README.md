# [TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety](https://arxiv.org/abs/2307.15311)
Large Language Models (LLMs) have shown remarkable effectiveness in various general-domain natural language processing (NLP) tasks. However, their performance in transportation safety domain tasks has been suboptimal, primarily attributed to the requirement for specialized transportation safety expertise in generating accurate responses. To address this challenge, we introduce TrafficSafetyGPT, a novel LLAMA-based model, which has undergone supervised fine-tuning using TrafficSafety-2K dataset which has human labels from government produced guiding books and ChatGPT-generated instruction-output pairs.

Ou Zheng<sup>1</sup>, Mohamed Abdel-Aty<sup>2</sup>, Dongdong Wang<sup>3</sup>, Chenzhu Wang<sup>4</sup>, Shengxuan Ding<sup>5</sup>

[![Custom badge](https://img.shields.io/badge/paper-Arxiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2307.15311)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/HUANGLIZI/ChatDoctor/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 

## Resources List

UCF Traffic Safety data from NSTHA Model Minimum Uniform Crash Criteria (MMUCC) Guideline Fourth edition, FHWA The Highway Safety Manual (HSM)[TrafficSafety-2K ](https://docs.google.com/spreadsheets/d/1PTztJw3pq1Eau0ZM2uL7N_yilv6H36QC/edit?usp=sharing&ouid=105044560872530659805&rtpof=true&sd=true).

Stanford Alpaca data for basic conversational capabilities. [Alpaca link](https://github.com/Kent0n-Li/ChatDoctor/blob/main/alpaca_data.json).

 ## How to fine-tuning
 

We fine-tune our models using standard Hugging Face training code.
We fine-tune LLaMA-7B with the following hyperparameters:

| Hyperparameter | LLaMA-7B | 
|----------------|----------|
| Batch size     | 128      | 
| Learning rate  | 2e-5     | 
| Epochs         | 3        |
| Max length     | 512      | 
| Weight decay   | 0        | 


 ```python
torchrun 
 ```
 
 
Fine-tuning with Lora 
```python

 ```
 
 ## How to inference
 You can build a ChatDoctor model on your own machine and communicate with it.
 ```python

 ```
## Acknowledgments

We would like to thank the authors and developers of the following projects, this project is built upon these great projects.

- [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)
- [Stanford_Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [LLaMa](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)

## Reference

TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety
```
@misc{zheng2023trafficsafetygpt,
      title={TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety}, 
      author={Ou Zheng and Mohamed Abdel-Aty and Dongdong Wang and Chenzhu Wang and Shengxuan Ding},
      year={2023},
      eprint={2307.15311},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
