# final_project_model_interface

This project demonstrates a Dockerized machine learning pipeline for training, evaluation, and deployment. The pipeline includes data preprocessing, model training, and saving the trained model for inference, ensuring a streamlined and reproducible workflow.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [License](#license)

---

## Project Overview

This project includes the following:
- **`src/inference.py`**: The main script to load the model, preprocess input data, and make predictions.
- **`src/model_loader.py`**: Utility script to load a trained model.
- **`src/data_preprocessor.py`**: Preprocessor for input data.
- **`models/`**: Directory containing the trained model (`trained_model_2025-01-10.joblib`).
- **`Dockerfile`**: Used to build and deploy the Docker image.
- **`pyproject.toml`**: Poetry configuration file.

---

## Prerequisites

Before running this project, make sure you have the following installed:
- **Python 3.12+**: [Install Python](https://www.python.org/downloads/)
- **Poetry**: [Install Poetry](https://python-poetry.org/docs/#installation)
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)

---

## Installation

### Local Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd final_project_model_interface

