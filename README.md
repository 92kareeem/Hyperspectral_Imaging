
# Hyperspectral Imaging

## Overview
This repository contains code for analyzing a hyperspectral imaging dataset. The code performs data preprocessing, model training, and evaluation using a 1D Convolutional Neural Network (CNN).

## Repository Structure
*   `main.ipynb` or `main.py`: The main script containing the code for data loading, preprocessing, model training, and evaluation.
*   `report.pdf` : A short report summarizing the preprocessing steps, insights from dimensionality reduction, model selection, training, and evaluation details, and key findings with suggestions for improvement.
*   `TASK-ML-INTERN.csv`: The hyperspectral imaging dataset.
*   `README.md`: This file.

## Instructions to Install Dependencies and Run the Code

### Dependencies
*   Python 3.8 or higher
*   Pandas
*   NumPy
*   Scikit-learn
*   TensorFlow
*   Matplotlib

### Installation

1.  Clone the repository:

    ```
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  Create a virtual environment (recommended):

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required packages:

    ```
    pip install pandas numpy scikit-learn tensorflow matplotlib
    ```

### Running the Code

1.  Ensure that the `TASK-ML-INTERN.csv` file is in the same directory as the script.

2.  Run the script:

    *   For Jupyter Notebook:

        ```
        jupyter notebook main.ipynb
        ```

    *   For Python script:

        ```
        python main.py
        ```

3.  The script will load the dataset, preprocess it, train the CNN model, evaluate the model, and display the evaluation metrics and a scatter plot of actual vs. predicted values.

## Brief Overview of the Code
The code performs the following steps:

1.  **Data Loading**: Loads the hyperspectral imaging dataset from a CSV file.
2.  **Data Preprocessing**:
    *   Inspects class distribution.
    *   Scales the labels.
    *   Removes classes with few samples.
    *   Removes the `hsi_id` column.
    *   Splits the dataset into training and testing sets.
    *   Reshapes the data for CNN input.
    *   Converts the data to `float32`.
3.  **Model Definition**: Defines a 1D CNN model for regression.
4.  **Model Training**: Trains the CNN model on the training data.
5.  **Model Evaluation**: Evaluates the trained model on the test data using regression metrics (MAE, RMSE, R² Score) and visualizes the results using a scatter plot.

