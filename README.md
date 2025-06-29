# README.md

# Cat and Dog Image Classification

This project implements a deep learning pipeline to classify images of cats and dogs using a ResNet18 model. The solution includes data loading, training, evaluation, and prediction scripts, as well as a Gradio web interface for easy image classification.

## Demo

![demo](<Screenshot 2025-06-30 012920.png>)

## Features

- Train a ResNet18 model on your own dataset of cat and dog images
- Evaluate model performance on test data
- Predict the class of new images using a trained model
- Interactive web demo using Gradio
- Jupyter notebooks for data exploration, training, and prediction

## Project Structure

- `app.py`: Gradio web interface for image classification
- `train.py`: Script to train the model
- `evaluate.py`: Script to evaluate the model
- `predict.py`: Script to predict on new images
- `models/`: Contains model architecture (ResNet18)
- `utils/`: Data loading utilities
- `data/`: Training and test image folders
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `test_images/`: Example images for testing

## Getting Started

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Prepare your dataset in `data/train` and `data/test` directories.

3. Train the model:
    ```sh
    python train.py
    ```

4. Launch the Gradio app:
    ```sh
    python app.py
    ```

## Requirements

See [requirements.txt](requirements.txt) for a full list of dependencies.

## License

This project is for educational purposes.