# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE, mutual_info_regression, SelectKBest, f_regression

# Load the Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Prepare the Data
target = 'medv'  # Median value of homes
X = data.drop(columns=[target])  # Feature variables
y = data[target]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store results for all feature selection methods
results_rfe = []
results_mift = []
results_selectkbest = []

# RFE
model = LinearRegression()
for num_features in range(1, X.shape[1] + 1):
    selector = RFE(estimator=model, n_features_to_select=num_features)
    selector.fit(X_train, y_train)
    selected_features = X.columns[selector.support_]

    model.fit(X_train[selected_features], y_train)
    predictions = model.predict(X_test[selected_features])
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    results_rfe.append((num_features, selected_features.tolist(), rmse, r2))

# MIFT
mi_scores = mutual_info_regression(X_train, y_train)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

for num_features in range(1, X.shape[1] + 1):
    top_features = mi_df.nlargest(num_features, 'Mutual Information')['Feature'].values
    
    model.fit(X_train[top_features], y_train)
    predictions = model.predict(X_test[top_features])
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    results_mift.append((num_features, top_features.tolist(), rmse, r2))

# SelectKBest
for num_features in range(1, X.shape[1] + 1):
    selector = SelectKBest(score_func=f_regression, k=num_features)
    selector.fit(X_train, y_train)
    selected_features = X.columns[selector.get_support()]

    model.fit(X_train[selected_features], y_train)
    predictions = model.predict(X_test[selected_features])
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    results_selectkbest.append((num_features, selected_features.tolist(), rmse, r2))

# Prepare the graph
num_features_range = np.arange(1, X.shape[1] + 1)
rmse_rfe = [result[2] for result in results_rfe]
rmse_mift = [result[2] for result in results_mift]
rmse_selectkbest = [result[2] for result in results_selectkbest]

plt.figure(figsize=(10, 6))
plt.plot(num_features_range, rmse_rfe, label='Recursive Feature Elimination', marker='o')
plt.plot(num_features_range, rmse_mift, label='Mutual Information', marker='o')
plt.plot(num_features_range, rmse_selectkbest, label='SelectKBest', marker='o')
plt.title('RMSE for Different Feature Selection Algorithms')
plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.savefig('static/rmse_plot.png')  # Save the plot
plt.close()

# Create Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', results_rfe=results_rfe, results_mift=results_mift, results_selectkbest=results_selectkbest)

if __name__ == '__main__':
    app.run(debug=True)
