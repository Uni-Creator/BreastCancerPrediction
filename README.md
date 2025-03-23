# Breast Cancer Prediction App
![GitHub Repo stars](https://img.shields.io/github/stars/Uni-Creator/BreastCancerPrediction?style=social)  ![GitHub forks](https://img.shields.io/github/forks/Uni-Creator/BreastCancerPrediction?style=social)

## 📌 Overview
The **Breast Cancer Prediction App** is a machine learning-based web application built using **Streamlit**. It allows users to input 30 feature values (or autofill a random row from the dataset) and predict whether the tumor is **Malignant** or **Benign** using multiple ML models. Additionally, the app displays the prediction confidence scores of each model and visualizes them using a bar chart.

## 🚀 Features
- **Multiple ML Models**: Supports **RandomForest, SVM, Logistic Regression, KNN, Decision Tree, and Naïve Bayes**.
- **Manual Input or Autofill**: Users can enter feature values manually or autofill with a random dataset row.
- **Prediction Confidence Scores**: Displays how confident each model is in its prediction.
- **Original Diagnosis Display**: If a random row is selected, it shows the actual diagnosis from the dataset.
- **Data Visualization**: A bar chart representing confidence scores of each model.

## 🏗️ Tech Stack
- **Python**
- **Streamlit** (for the web interface)
- **Pandas & NumPy** (for data processing)
- **Matplotlib** (for visualization)
- **Scikit-learn** (for ML models)

## 📂 Project Structure
```
BreastCancerPrediction/
│── models/                   # Pre-trained ML models
│   ├── model_RandomForest.pkl
│   ├── model_SVM.pkl
│   ├── model_LogisticRegression.pkl
│   ├── model_KNN.pkl
│   ├── model_DecisionTree.pkl
│   ├── model_NaiveBayes.pkl
│── breast-cancer.csv          # Dataset
│── main.py                    # Streamlit application
│── README.md                  # Project documentation
```

## 📦 Installation & Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/BreastCancerPrediction.git
   cd BreastCancerPrediction
   ```
2. **Install dependencies**
   ```sh
   pip install streamlit sklearn seaborn matplotlib pandas
   ```
3. **Run the application**
   ```sh
   streamlit run main.py
   ```

## 📊 How It Works
1. Enter the 30 feature values manually or **click "Autofill Random Row"**.
2. Choose a machine learning model from the dropdown.
3. Click **"Predict"** to get the tumor classification (**Malignant** or **Benign**).
4. View **confidence scores** of all models and the **bar chart visualization**.
5. If autofill is used, compare predictions with the **original diagnosis**.

## 🛠️ Future Improvements
- Add more ML models for better comparison.
- Implement an **Explainable AI (XAI)** module to show feature importance.
- Deploy the app using **Streamlit Cloud** or **Heroku**.

## 🤝 Contributing
Contributions are welcome! Feel free to open an **issue** or submit a **pull request**.

## 📄 License
This project is licensed under the **Apache License 2.0**.

---
