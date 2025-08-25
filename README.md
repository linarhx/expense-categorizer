# Expense Categorizer

A machine learning application that automatically categorizes financial transactions using natural language processing.

## Features

- **Automatic Categorization**: Uses TF-IDF vectorization and logistic regression to categorize transactions
- **Web Interface**: Streamlit-based web application for easy interaction
- **Pre-trained Model**: Comes with a pre-trained model for immediate use

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

```bash
cd src
python train.py
```

### Running the Web App

```bash
cd app
streamlit run main.py
```

## Data Format

The model expects CSV files with two columns:
- `transaction`: Text description of the transaction
- `category`: The category label

## Model

The system uses a pipeline combining:
- TF-IDF vectorization for text feature extraction
- Logistic regression for classification

## Project Structure

```
expense_categorizer/
├── app/           # Streamlit web application
├── data/          # Training and sample data
├── models/        # Trained model files
├── src/           # Training scripts
└── requirements.txt
```
