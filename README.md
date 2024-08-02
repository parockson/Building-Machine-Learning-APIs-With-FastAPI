# Building-Machine-Learning-APIs-With-FastAPI

## Sepsis Detection App

### Overview

This Sepsis Detection App is designed to predict sepsis in patients based on various medical and demographic features. The app leverages machine learning models and is built using FastAPI and Streamlit. Docker support will be added soon to facilitate containerization and deployment.

### Table of Contents

1. [Project Structure](#project-structure)
2. [CRISP-DM Framework](#crisp-dm-framework)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

### Project Structure

```
├── app
│   ├── main.py            # FastAPI application
│   ├── models.py          # Machine learning models
│   └── utils.py           # Utility functions
├── streamlit
│   └── app.py             # Streamlit application
├── data
│   ├── raw                # Raw data files
│   └── processed          # Processed data files
├── models
│   └── trained_models     # Trained machine learning models
├── notebooks
│   └── exploration.ipynb  # Jupyter notebook for data exploration
├── requirements.txt       # Python dependencies
└── README.md              # Project readme file
```

### CRISP-DM Framework

The development of this app follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, ensuring a structured and well-organized approach to data science projects.

1. **Business Understanding**
   - Objective: Predict sepsis in patients to provide timely and appropriate medical interventions.
   - Goals: Improve early detection of sepsis, reduce mortality rates, and enhance patient outcomes.

2. **Data Understanding**
   - Data Sources: The dataset includes medical and demographic information such as plasma glucose levels (PRG), blood pressure (PR), various blood work results (PL, SK, TS, BD2), BMI (M11), and age.
   - Exploratory Data Analysis: Conducted to understand the distribution and relationships between features and the target variable (sepsis).

3. **Data Preparation**
   - Data Cleaning: Handled missing values, outliers, and inconsistencies.
   - Feature Engineering: Created new features and transformed existing ones to improve model performance.
   - Data Splitting: Split the data into training and testing sets.

4. **Modeling**
   - Algorithms Used: RandomForestClassifier, SVC, XGBClassifier, and LGBMClassifier.
   - Model Training: Trained multiple models to identify the best performing ones.
   - Hyperparameter Tuning: Performed to optimize model performance.

5. **Evaluation**
   - Metrics: Evaluated models using precision, recall, F1-score, and AUC-ROC due to class imbalance.
   - Model Selection: Selected the top 3 models based on evaluation metrics.

6. **Deployment**
   - FastAPI: Used to build the backend API for model inference.
   - Streamlit: Used to build the frontend for user interaction and visualization.
   - Docker: Containerization support will be added soon for easier deployment.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sepsis-detection-app.git
   cd sepsis-detection-app
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

### Usage

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run streamlit/app.py
   ```

3. Access the app in your browser:
   - FastAPI: `http://127.0.0.1:8000`
   - Streamlit: `http://localhost:8501`

### Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Create a pull request.

### License

This project is licensed under the MIT License.

---

Feel free to reach out with any questions or feedback regarding this project. Your contributions and suggestions are highly appreciated!