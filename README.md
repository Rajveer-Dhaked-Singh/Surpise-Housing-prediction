#  House Price Prediction — Ridge, Lasso & Gradient Boosting

A supervised machine learning project to predict residential property sale prices using the [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The project covers end-to-end ML pipeline development: data cleaning, exploratory data analysis, feature engineering, regularized regression models, and a gradient boosting model for comparison.

---

## Problem Statement

Predict the final sale price of homes based on 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa. This is a regression problem where the target variable (`SalePrice`) is continuous.

---

## Project Structure

```
├── main.py                  # Full pipeline: EDA → preprocessing → modelling → evaluation
├── train.csv                # Raw training dataset (from Kaggle)
└── README.md
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| ML models | `scikit-learn` |
| Boosting | `sklearn.ensemble.GradientBoostingRegressor` |

---

##  Workflow

### 1. Data Cleaning
- Identified and filled missing values for 17 columns (e.g., `PoolQC`, `GarageType`, `BsmtQual`) with `"None"` to preserve categorical meaning
- Converted `LotFrontage` and `MasVnrArea` to numeric; remaining nulls filled with column mean/mode
- Dropped the `Id` column (non-informative)
- Cast `MSSubClass`, `OverallQual`, `OverallCond` as object types (ordinal, not continuous)

### 2. Target Variable Transformation
- `SalePrice` was right-skewed (skewness > 1)
- Applied **log transformation** (`np.log`) to normalise the distribution before modelling

### 3. Feature Engineering
- Created `Age = YrSold − YearBuilt` to capture property age as a single feature
- Dropped `YrSold` and `YearBuilt` after deriving `Age`

### 4. Exploratory Data Analysis (EDA)
- **Univariate**: Boxplots for numerical columns; pie charts for categorical columns
- **Bivariate**: Bar plots of `HouseStyle`, `BldgType`, `BsmtQual` vs `SalePrice`
- **Correlation analysis**: Full heatmap + top-10 correlated feature heatmap
- **Pairplot** on top predictors: `OverallQual`, `GrLivArea`, `GarageArea`, etc.

Key findings:
- `GarageArea` and `GarageCars` are highly correlated (0.88)
- `GrLivArea` and `TotRmsAbvGrd` are highly correlated (0.83)
- `TotalBsmtSF` and `1stFlrSF` are highly correlated (0.82)

### 5. Data Preparation
- One-hot encoded all categorical columns using `pd.get_dummies(drop_first=True)`
- Train/test split: **70% train / 30% test** (`random_state=28`)
- Scaled numerical features using `StandardScaler` — fitted on train set only, applied to test set

### 6. Model Building

#### Ridge Regression
- Hyperparameter tuned via `GridSearchCV` (5-fold CV, 28 alpha values)
- Best alpha: **500**
- Train R²: **0.94** | Test R²: **0.79** | RMSE: **0.18**

#### Lasso Regression
- Hyperparameter tuned via `GridSearchCV` (5-fold CV, same alpha grid)
- Best alpha: **0.01**
- Train R²: **0.92** | Test R²: **0.76** | RMSE: **0.20**
- Lasso eliminated several irrelevant features (set coefficients to exactly 0)

#### Gradient Boosting Regressor
- `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`
- Train R²: **0.99** | Test R²: **0.89** | RMSE: **0.13**

### 7. Model Comparison

| Model | Train R² | Test R² | RMSE (Test) |
|---|---|---|---|
| Ridge | 0.94 | 0.79 | 0.18 |
| Lasso | 0.92 | 0.76 | 0.20 |
| Gradient Boosting | 0.99 | 0.89 | 0.13 |

### 8. Residual Analysis
- Scatter plot of predicted values vs residuals confirms no strong systematic bias
- Residuals approximately centred around 0

### 9. Cross-Validation
- 5-fold cross-validation on Ridge model confirms generalisation:
  `CV R² (mean ± std): ~0.81 ± 0.03`

---

## Feature Importance

After exponentiating coefficients (inverse log) to interpret on original price scale:

**Top features (Ridge):** `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `Age`

**Lasso feature selection:** Eliminated low-signal dummy features while preserving core structural and quality predictors.

---

##  How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Download the dataset**
   - Get `train.csv` from [Kaggle: House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
   - Place it in the project root directory

4. **Run the script**
   ```bash
   python main.py
   ```
   Or open in Jupyter/Google Colab and run all cells.

---

##  Key Learnings

- Log-transforming a skewed target variable significantly improves linear model performance
- Lasso regression acts as an automatic feature selector by pushing irrelevant coefficients to zero
- Ridge regression retains all features but shrinks their influence — useful when multicollinearity is present
- Gradient Boosting outperforms regularised linear models on this dataset by capturing non-linear relationships
- Proper train/test separation during scaling is critical to prevent data leakage

---

##  Possible Extensions

- Add `XGBoost` or `LightGBM` for further performance gains
- Apply `StratifiedKFold` with price bins for more robust evaluation
- Use `SHAP` values for model explainability
- Build a prediction API with `FastAPI` or `Flask`
- Deploy on Streamlit for interactive property price estimation

---

##  Dataset Source

Kaggle — [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

> Dataset by Dean De Cock. Contains 1,460 training samples with 79 features describing residential properties in Ames, Iowa.
