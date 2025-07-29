# Stock Price Prediction Project

## Overview
This project implements a stock price prediction model using historical stock price data. It utilizes machine learning classifiers, specifically Support Vector Machine (SVM) and K-Nearest Neighbors (KNN), to predict the direction of stock prices (up or down). The project includes data preprocessing, model training, evaluation metrics, and visualizations of the results.

## Files
- **price.py**: Contains the main code for the stock price prediction model, including data preprocessing, model training, evaluation metrics, and visualizations.
- **stock_price_direction_data.csv**: The dataset used for training and testing the models, containing historical stock price data and a target variable indicating price direction.
- **.gitignore**: Specifies files and directories to be ignored by Git, such as Python bytecode files and virtual environment directories.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd stock-price-prediction
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Ensure that the dataset `stock_price_direction_data.csv` is in the project directory.
2. Run the main script:
   ```
   python price.py
   ```
3. The script will output classification reports, accuracy scores, and visualizations of the model performance.

## Dataset
The dataset `stock_price_direction_data.csv` contains historical stock prices along with a target variable that indicates whether the price will go up or down. The features used for prediction are derived from the stock price data.

## Models
- **Support Vector Machine (SVM)**: A powerful classifier that works well for high-dimensional spaces.
- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm that classifies based on the majority class of the nearest neighbors.

## Visualizations
The project includes visualizations such as confusion matrices for model evaluation and a correlation heatmap to understand feature relationships.

## License
This project is licensed under the MIT License - see the LICENSE file for details.