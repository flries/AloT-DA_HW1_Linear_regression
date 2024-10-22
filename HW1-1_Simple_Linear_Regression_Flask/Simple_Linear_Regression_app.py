from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Generate synthetic data
def generate_data(a=2, b=5, noise=1, num_points=100):
    np.random.seed(0)  # For reproducibility
    X = np.random.uniform(0, 10, num_points)  # Random X values
    y = a * X + b + np.random.normal(0, noise, num_points)  # y = ax + b + noise
    return X.reshape(-1, 1), y

# Train the model
def train_model(a, b, noise, num_points):
    X, y = generate_data(a, b, noise, num_points)
    
    # Create and train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict and evaluate
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    return X, y, y_pred, mse

# Plot regression
def plot_regression(X, y, y_pred, a, b, mse):
    plt.figure(figsize=(8,6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, y_pred, color='red', label=f'Predicted Line: y = {a}x + {b}')
    plt.title(f'Linear Regression (MSE: {mse:.2f})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')

# Flask route to render the form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        a = float(request.form['a'])
        b = float(request.form['b'])
        noise = float(request.form['noise'])
        num_points = int(request.form['num_points'])
        
        X, y, y_pred, mse = train_model(a, b, noise, num_points)
        plot_url = plot_regression(X, y, y_pred, a, b, mse)
        
        return render_template('index.html', plot_url=plot_url, a=a, b=b, noise=noise, num_points=num_points, mse=mse)
    
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
