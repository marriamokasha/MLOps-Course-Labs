# 🧪 Research: Bank Customer Churn Prediction

This branch focuses on experimenting with multiple machine learning models to predict customer churn in a bank. It uses MLflow for tracking experiments and storing models and metrics.

## 📊 Objective

To compare Logistic Regression, Random Forest, and Gradient Boosting models for predicting customer churn and select the best-performing model based on F1 Score.

## 📁 Project Structure (Research Branch)

<pre lang="markdown"><code> ```
  .
  ├── dataset/ 
  │ └── Churn_Modelling.csv 
  ├── mlruns/ 
  │ └── [MLflow tracking artifacts] 
  ├── models/ 
  │ ├── best_model.pkl 
  │ └── preprocessor.pkl 
  ├── src/ 
  │ └── train.py 
  ├── best_model_info.txt 
  ├── requirements.txt 
  └── README.md 
``` </code></pre>



## 🚀 How to Run

1. **Clone the Repository**
    ```bash
    git clone https://github.com/marriamokasha/bank-churn-prediction.git
    cd bank-churn-prediction
    ```

2. **Create & Activate a Virtual Environment**
    ```bash
    python -m venv churn_prediction
    churn_prediction\Scripts\activate   # On Windows
    ```

3. **Install +dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Start MLflow tracking server**
    ```bash
    mlflow ui
    ```
    Access it at: [http://localhost:5000](http://localhost:5000)

5. **Run the experiment**
    ```bash
    python src/train.py
    ```

6. **Outputs**
    - Best model and preprocessor saved in `/models/`
    - Best run details written to `best_model_info.txt`
    - MLflow logs all experiments in `/mlruns/`

## ✅ Best Model

- **Model:** GradientBoostingClassifier  
- **F1 Score:** 0.7651  
- **Accuracy:** 0.7735  
- **Precision:** 0.7803  
- **Recall:** 0.7504  
