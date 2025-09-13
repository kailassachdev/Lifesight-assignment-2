# Marketing Mix Model for Revenue Prediction

This project analyzes a 2-year weekly dataset of paid media metrics, direct response levers, and revenue to build a machine learning model that explains revenue as a function of the input variables.

## Project Structure

- `Train/`: Contains the raw dataset.
- `plots/`: Contains all the generated plots for analysis and diagnostics.
- `frontend/`: Contains a simple frontend to visualize the results.
- `01_initial_eda.py`: Script for initial exploratory data analysis.
- `02_data_prep_and_viz.py`: Script for data preparation and visualization.
- `03_modeling.py`: The main script for feature engineering, modeling, and evaluation.
- `revenue_drivers.csv`: A CSV file containing the coefficients of the final model, indicating the main drivers of revenue.
- `README.md`: This file.

## How to Run

1. **Prerequisites**: Make sure you have Python installed, along with the following libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
   You can install them using pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Run the scripts in order**:
   ```bash
   python 01_initial_eda.py
   python 02_data_prep_and_viz.py
   python 03_modeling.py
   ```

## Frontend Visualization

This project includes a simple frontend to visualize the results of the analysis. To view it, open the `frontend/index.html` file in your web browser.

The frontend displays:
- The key performance metric (RMSE).
- All the generated plots.
- A table of the revenue drivers, loaded dynamically from the `revenue_drivers.csv` file.

## Methodology

### 1. Data Preparation

- The `week` column was converted to a datetime index.
- The target variable, `revenue`, was found to be highly skewed. A `log(revenue + 1)` transformation was applied to make its distribution more normal.
- Feature scaling was performed using `StandardScaler` to handle features with different scales.

### 2. Feature Engineering

- **Time-based features**: `time_trend`, `month`, and `week_of_year` were created to capture trend and seasonality.
- **Lagged features**: 4-week lags were created for all spend and direct response variables to account for the delayed effect of marketing activities.

### 3. Modeling and Causal Framing

A two-stage modeling approach was used to address the causal assumption that Google spend is a mediator between other social media channels and revenue.

- **Stage 1: Modeling Google Spend**: A Ridge regression model was trained to predict `google_spend` based on other marketing channels (Facebook, TikTok, Instagram, Snapchat) and other features. The *residuals* from this model were then calculated. These residuals represent the portion of Google spend that is *not* influenced by the other social media channels.

- **Stage 2: Modeling Revenue**: A second Ridge regression model was trained to predict `log(revenue + 1)`. The features included all variables, but the original `google_spend` was replaced with the **residuals** from the Stage 1 model. This allows for a more direct interpretation of the impact of the uninfluenced portion of Google spend on revenue.

### 4. Validation

A time-series-aware train-test split was used (first 80% for training, last 20% for testing) to prevent data leakage from the future. The hyperparameters for the Ridge models were tuned using cross-validation on the training set.

## Results and Insights

- **Model Performance**: The final model has an **RMSE of 78,577.76** on the test set.
- **Key Revenue Drivers**: The most influential drivers of revenue, based on the model coefficients, are:
    - **`instagram_spend`**: The strongest positive driver.
    - **`average_price`**: The strongest negative driver (higher price is associated with lower revenue).
    - **`sms_send`**: A strong positive driver.
    - **Lagged Facebook spend**: Spend from 2 weeks prior has a significant positive impact.
    - **`google_spend_residuals`**: The uninfluenced portion of Google spend has a positive impact on revenue, confirming its role as a direct driver when its mediated effect is controlled for.

- **Diagnostics**:
    - The `actual_vs_predicted.png` plot shows that the model captures the general trend of revenue but struggles with the very high spikes, which is expected given the high volatility.
    - The `residuals_distribution.png` plot shows that the errors are mostly centered around zero, but there are a few large errors corresponding to the revenue spikes the model missed.

## Conclusion

This project successfully developed a machine learning model to explain revenue, while also accounting for the complex causal relationships between marketing channels. The insights from this model can be used by a growth/marketing team to make more informed decisions about budget allocation and pricing strategies.