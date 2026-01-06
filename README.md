# DNN
OpenStory++ Mini Storytelling Evaluation Project

Author: Palem Ravi

Introduction and Problem Statement

This repository contains the final assessment project for a storytelling inference and evaluation task using the OpenStory++ dataset from MAPLE-WestLake-AIGC. The objective of this project is to design a lightweight storytelling evaluation pipeline under restricted computational resources.

The core focus is visual–text storytelling, where a model must interpret image–caption pairs and assess the quality of generated narrative descriptions. Due to hardware limitations, the project is intentionally constrained to a small experimental setup while still demonstrating dataset handling, evaluation methodology, and reasoning analysis.

Only 20 samples from the dataset are used to simulate a minimal inference pipeline and compute baseline BLEU-score performance.

Methods
System Overview

The project implements a simplified storytelling evaluation workflow consisting of:

dataset loading and preprocessing

minimal inference simulation

baseline BLEU score computation

structured results export and visualisation

The goal is not to improve storytelling model performance, but rather to illustrate the evaluation process in a controlled environment.

A high-level diagram of the pipeline is shown here:

results/sample_visualization.png

Dataset Description

We use the OpenStory++ dataset:

Source: https://huggingface.co/datasets/MAPLE-WestLake-AIGC/OpenstoryPlusPlus

Domain: visual–caption storytelling scenes

Subset: 20 records (memory-restricted)

The dataset primarily consists of woodworking-related environments and human activities.

Results
Quantitative Evaluation

BLEU score is used as the baseline performance metric.

Visual summaries include:

BLEU score chart → results/bleu_chart.png

Full results table → results/full_results_table.csv

Qualitative Analysis

Example storytelling outputs and visual references are included in:

results/sample_visualization.png

results/sample_prediction.txt

These provide insight into narrative alignment with reference captions.

Conclusions

This mini-scale storytelling evaluation pipeline demonstrates that even under strong hardware constraints, it is possible to:

load and process multimodal narrative data

simulate storytelling inference

compute quantitative language metrics

export structured experimental results

The project highlights foundational storytelling evaluation concepts rather than model sophistication.

Future Work

Potential extensions include:

scaling to larger sample sizes

incorporating real generative models

testing additional evaluation metrics

analysing cross-modal reasoning depth

benchmarking alternative datasets


testing additional evaluation metrics

analysing cross-modal reasoning depth

benchmarking alternative datasets
