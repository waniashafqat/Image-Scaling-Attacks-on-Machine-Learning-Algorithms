# Image Scaling Attacks on Machine Learning Algorithms:

## Table of Contents
- [Overview](#overview)
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Installation](#installation)
- [Usage](#usage)
- [Attack Analysis](#attack-analysis)
- [Defenses](#defenses)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This repository contains the code and resources for the research project on image scaling attacks targeting machine learning (ML) algorithms, particularly focusing on traffic sign recognition systems. The project investigates the vulnerabilities introduced by image scaling in ML models and proposes defense mechanisms to mitigate these attacks.

## Introduction
This project explores the susceptibility of ML algorithms to image scaling attacks, a type of adversarial attack that manipulates the size and resolution of input images to induce incorrect model predictions. The research focuses on traffic sign recognition systems using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It aims to improve the robustness of these systems against such adversarial manipulations and test adversarial attack strategies on neural network-based classifiers, primarily Convolutional Neural Networks (CNNs) built using Keras.

## Project Structure
- `Pictures/`: Contains the pictures used for attacking.
- `GTSRB model/`: Contains the CNN Keras model trained on the GTSRB dataset.
- `(01) GTSRB Model/`: Jupyter notebook for data analysis, model training, and evaluation.
- `(02) Interpolations/`: Jupyter notebook for finding vulnerable interpolations
- `(03) Image Scaling Attacks/`: Jupyter notebook for implementing image scaling attacks.
- `(04) Defenses/`: Jupyter notebook for defense mechanisms for image scaling attacks.
- `README.md`: Project overview and instructions.

## Setup
### Prerequisites
- Python 3.12 or higher
- IDE: Jupyter Notebook
- Required libraries: OpenCV, Pillow, TensorFlow, Keras, NumPy
- OS: Windows 10/11, Ubuntu 20.04 LTS, or other Linux distributions.

### Installation
To set up the environment for this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/waniashafqat/Image-Scaling-Attacks-on-Machine-Learning-Algorithms.git
   cd Image-Scaling-Attacks-on-Machine-Learning-Algorithms
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

## Usage
### Data Preparation and Model Training
1. Download and extract the GTSRB dataset from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
2. Preprocess the dataset and train the model using the `GTSRB Model.ipynb`file.

### Data Preparation and Model Training
1. For finding vulnerabilities in ML model run ‘Interpolations.ipynb’

### Attack
1. Implement image scaling attacks on GTSRB model using ‘Image Scaling Attacks.ipynb’
### Defense
1. For defense mechanisms use  ‘Defenses.ipynb’

## Attack Analysis
The attack analysis includes:
- Dataset and Model: Utilizes the GTSRB dataset and Keras-based CNN models.
- Attack Design: Focuses on creating adversarial images using image scaling techniques.
- Perturbations and Norms: Implements L0, L2, and L∞ norms to generate minimal but effective perturbations.
- Interpolation Techniques: Employs various interpolation methods to understand their impact on attack effectiveness.

## Defenses
Several defense mechanisms are proposed to counteract image scaling attacks:
- Pixel-wise Difference
- Structural Similarity Index (SSIM)
- Color Histogram-Based Detection
- Robust Scaling Algorithms

## Limitations
- The findings are primarily based on the GTSRB dataset and specific models developed with Keras.
- Different architectures or frameworks might exhibit varying levels of susceptibility to image scaling attacks.
- The practicality of deploying our defense mechanisms in real-world applications needs further evaluation.

## Conclusion
This research highlights the vulnerabilities of ML models to image scaling attacks and proposes practical solutions to enhance their robustness. By addressing these vulnerabilities, we can improve the security and reliability of ML systems in critical applications like autonomous driving.

For more detailed information, please refer to the research paper included in this repository.

## Contributing
We welcome contributions to improve the project. Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact
For any questions or feedback, please contact [waniashafqat02@gmail.com](mailto:waniashafqat02@gmail.com).
