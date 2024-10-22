# Will LLMs Replace the Encoder-Only Models in Temporal Relation Classification?
<p align="center">
    <a href="https://arxiv.org/pdf/2410.10476v1">
        <img alt="paper" src="https://badgen.net/badge/EMNLP 2024/Accepted/green?icon=awesome">
    </a>
    <a href="https://arxiv.org/pdf/2410.10476v1">
        <img alt="paper" src="https://badgen.net/badge/arXiv/online/yellow">
    </a>
    <a href="https://github.com/BrownFortress/LLMs-TRC/blob/main/LICENSE">
        <img alt="License" src="https://badgen.net/static/license/MIT/blue">
    </a>
    <br/>
</p>

***TL;DR***
We test the performance of LLMs in the Temporal Relation Classification task by using ICL and fine-tuning strategies. Then, we employ XAI techniques to understand why the RoBERTa-based model still outperforms LLMs. This analysis suggests the reason is the different pre-training tasks, i.e. masked language modelling vs causal language modelling. 

## Abstract
The automatic detection of temporal relations among events has been mainly investigated with encoder-only models such as RoBERTa. Large Language Models (LLM) have recently shown promising performance in temporal reasoning tasks such as temporal question answering. Nevertheless, recent studies have tested the LLMs' performance in detecting temporal relations of closed-source models only, limiting the interpretability of those results. In this work, we investigate LLMs' performance and decision process in the Temporal Relation Classification task. First, we assess the performance of seven open and closed-sourced LLMs experimenting with in-context learning and lightweight fine-tuning approaches. Results show that LLMs with in-context learning significantly underperform smaller encoder-only models based on RoBERTa. Then, we delve into the possible reasons for this gap by applying explainable methods. The outcome suggests a limitation of LLMs in this task due to their autoregressive nature, which causes them to focus only on the last part of the sequence. Additionally, we evaluate the word embeddings of these two models to better understand their pre-training differences. 

## Getting started
The repository is organized into four folders, namely:
-   `data_formatter`:
-   `encoder_architecture`:
-   `ICL_and_FT`:
-   `XAI_analysis`: