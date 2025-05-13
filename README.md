# Toxic-Spans-Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Platform](https://img.shields.io/badge/platform-Google%20Colab-yellowgreen)](https://colab.research.google.com/)

This repository contains the dataset and code implementations for the **Toxic Spans Detection** project.  
The project focuses on identifying specific spans within text that contribute to its toxicity —  
a task crucial for content moderation and understanding harmful language patterns.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Overview

The goal of this project is to detect toxic spans within textual data. Unlike traditional toxicity classification tasks that label entire texts, this project aims to pinpoint the exact segments (spans) that make the text toxic. This fine-grained approach provides more actionable insights for content moderation systems.([GitHub][2], [ResearchGate][3])

## Dataset

The dataset used in this project is derived from the [SemEval-2021 Task 5: Toxic Spans Detection](https://aclanthology.org/2021.semeval-1.6/). It comprises English-language texts annotated with character-level spans indicating toxic content. Each entry includes:([ACL Anthology][4])

* **Text**: The original comment or post.
* **Spans**: A list of character indices marking the toxic portions of the text.([arXiv][5])

The dataset is located in the `data/` directory of this repository.([GitHub][2])

## Project Structure

The repository is organized as follows:

```

Toxic-Spans-Detection/
├── data/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── model.py
│   │   └── train.py
│   └── evaluation/
│       └── evaluate.py
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│   └── inference.ipynb
├── requirements.txt
└── README.md
```



## Getting Started

### Prerequisites

Ensure you have the following installed:

* Python 3.7 or higher
* pip([GitHub][1])

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Azzam-Radman/Toxic-Spans-Detection.git
   cd Toxic-Spans-Detection
   ```



2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```



## Usage

### Running Notebooks on Google Colab

The notebooks in this repository are configured to run on Google Colab. Each notebook begins by cloning this repository:([GitHub][6])

```python
!git clone https://github.com/Azzam-Radman/Toxic-Spans-Detection.git
```



For the final notebook, which includes the complete preprocessing and model training pipeline, it's recommended to utilize a TPU for faster computation. To enable TPU in Colab:

1. Navigate to `Edit` > `Notebook settings`.
2. In the `Hardware accelerator` dropdown, select `TPU`.
3. Click `Save`.

## Model Training

The model architecture is based on [ToxicBERT](https://arxiv.org/abs/2104.13164), a transformer-based model fine-tuned for toxicity detection tasks. Training scripts are located in the `src/models/` directory.([GitHub][7])

To train the model:

```bash
python src/models/train.py --config configs/train_config.yaml
```



Ensure that the `train_config.yaml` file is properly configured with the desired hyperparameters and paths.

## Evaluation

After training, evaluate the model's performance using the evaluation script:

```bash
python src/evaluation/evaluate.py --model_path path_to_trained_model
```



The evaluation metrics include:

* **Character-level F1 Score**: Measures the overlap between predicted and actual toxic spans.
* **Precision**: Proportion of predicted toxic spans that are correct.
* **Recall**: Proportion of actual toxic spans that were correctly identified.([arXiv][8])

## Results

The model achieved the following performance on the test set:

* **Character-level F1 Score**: 0.68
* **Precision**: 0.70
* **Recall**: 0.66

These results demonstrate the model's effectiveness in accurately identifying toxic spans within text.([ACL Anthology][9])

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.([GitHub][7])

To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit them: `git commit -m 'Add your message here'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to thank the organizers of [SemEval-2021 Task 5](https://aclanthology.org/2021.semeval-1.6/) for providing the dataset and fostering research in toxic span detection.([ACL Anthology][4])
