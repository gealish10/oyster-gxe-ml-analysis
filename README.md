# Machine Learning Analysis for Tetraploid Oyster G×E Interactions

## Overview

This repository contains the machine learning analysis code for evaluating genotype-by-environment (G×E) interactions in allotetraploid and autotetraploid oysters. The analysis employs ensemble machine learning models to predict growth performance and identify key genetic and environmental factors.

**Associated Publication:**  
Establishment of six types of allotetraploids derived from Crassostrea gigas and C. angulata and their breeding potential

**Authors:**  
Xianchao Bai et al.

**Institution:**  
Key Laboratory of Mariculture, Ministry of Education, Ocean University of China, Qingdao 266003, China
Qingdao Institute of Blue Seed Industry, Qingdao 266073, China;
Laboratory for Marine Fisheries Science and Food Production Processes, Qingdao National Laboratory for Marine Science and Technology, Qingdao 266237, China.
---

## Repository Contents

- `oyster_ml_analysis.py` - Main analysis script with complete workflow
- `example-data.xlsx` - Example dataset (360-day whole weight measurements)
- `README.md` - This documentation file
- `example-data_README.md` - Detailed documentation for the example dataset
- `requirements.txt` - Python package dependencies (optional)

---

## Requirements

### Python Version
- Python 3.11 or higher

### Required Packages
```bash
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
openpyxl>=3.0.0  # For reading Excel files
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```

---

## Quick Start

### Try with Example Data

The easiest way to get started is using the provided example dataset:

```python
# Clone the repository
git clone https://github.com/[your-username]/oyster-gxe-analysis.git
cd oyster-gxe-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl

# Run analysis on example data
python oyster_ml_analysis_final.py
```

**Note:** The script uses `example-data.xlsx` by default. To use your own data, modify the `file_path` variable in the `if __name__ == "__main__"` block.

**Expected output:**
- Sample size: 3,117 individuals
- Best model: Extra Trees (Test R² ≈ 0.4661)
- Top feature: complexity_temperature (importance ≈ 0.0995)

See `EXAMPLE_DATA_README.md` for detailed information about the example dataset.

---

## Data Format

### Input Data Requirements

The script expects an Excel file (`.xlsx`) with the following structure:

**Sheet**: "Sheet1"  
**Header**: Row 2 (index 1)  
**Data**: Rows 3-93 (indices 2-92)

**Column Organization:**
- Each genotype occupies 5 columns: `[Index, Environment1, Environment2, Environment3, Empty]`
- 12 genotypes × 5 columns = 60 total columns
- Genotype order: NGA, BGA, GGA, NAG, BAG, GAG, NGG, NAA, BGG, BAA, GGG, GAA

**Genotype Nomenclature:**
- First letter: Shell color group (N=Normal, B=Black, G=Golden)
- Last two letters: Cross type
  - GG = C.gigas♀ × C.gigas♂ (autotetraploid)
  - GA = C.gigas♀ × C.angulata♂ (forward hybrid)
  - AG = C.angulata♀ × C.gigas♂ (reverse hybrid)
  - AA = C.angulata♀ × C.angulata♂ (autotetraploid)

**Environments:**
- rc: Rongcheng (Temperature: 11.60°C)
- rs: Rushan (Temperature: 12.72°C)  
- jn: Jiaonan (Temperature: 13.64°C)

**Measurements:**
- Each cell contains growth measurements (whole weight in g)
- 90 individuals per genotype-environment combination
- Total sample size after quality control: 3117 individuals

---

## Usage

### Basic Usage

```python
# Run the complete analysis
python oyster_ml_analysis.py
```

### Custom Usage

```python
from oyster_ml_analysis import main

# Specify your data file path
file_path = "path/to/your/data.xlsx"

# Run analysis (plots displayed interactively)
results = main(file_path)

# Or save plots to directory
results = main(file_path, plot_output_dir="./output_plots")
```

### Accessing Results

```python
# Best model information
best_model = results['best_model']
best_model_name = results['best_model_name']

# Model performance comparison
performance_df = results['results_df']
print(performance_df)

# Feature importance rankings
importance_df = results['feature_importance']
print(importance_df.head(10))

# Processed data
data_long = results['data_long']
feature_cols = results['feature_cols']
```

---

## Analysis Pipeline

### Step 1: Data Loading and Transformation
- Load Excel data from wide format
- Transform to long format (one row per individual)
- Extract growth measurements for each genotype-environment combination
- Apply outlier removal using 3-sigma rule
- Log transformation for normality: `log(growth + 1)`

### Step 2: Feature Engineering

**Base Features (12):**
- `temperature`: Environmental temperature (°C)
- `env_rank`: Ordinal environment rank (1=rc, 2=rs, 3=jn)
- `nutrition_rank`: Food availability proxy (1=low, 3=high)
- `is_hybrid`: Binary hybrid indicator (0=purebred, 1=hybrid)
- `hybrid_forward`: C.gigas♀ × C.angulata♂ crosses (GA)
- `hybrid_reverse`: C.angulata♀ × C.gigas♂ crosses (AG)
- `genetic_complexity`: Genetic background (1=purebred, 2=hybrid)
- `is_major_cross`: Major hybrid crosses indicator
- `genetic_group_N/B/G`: Shell color group indicators (one-hot encoded)
- `geno_env_interaction`: Hybrid status × environment rank

**Enhanced Features (11):**
- **Polynomial**: `temperature²`, `log(temperature)`
- **Two-way interactions**: 
  - `hybrid×temperature`
  - `complexity×temperature`
  - `complexity×env_rank`
  - `nutrition×temperature`
  - `complexity×nutrition`
  - `hybrid×nutrition`
- **Three-way interactions**: 
  - `complexity×temperature×nutrition`
  - `hybrid×temperature×nutrition`
- **Composite index**: `environment_index = (temperature + 2×nutrition)/3`

**Total: 23 features**

### Step 3: Data Preprocessing
- **Train-test split**: 80-20 stratified by environment
  - Training set: ~2,497 samples
  - Test set: ~625 samples
- **Feature standardization**: Continuous features scaled to mean=0, std=1
  - Standardized: `temperature`, `env_rank`, `temperature²`, `log(temperature)`
  - Unchanged: Binary and categorical features
- **Random seed**: 42 for reproducibility

### Step 4: Model Training and Comparison

**Models Evaluated:**
1. **Random Forest (RF)**: Bootstrap aggregating of decision trees
2. **Gradient Boosting (GB)**: Sequential boosting with gradient descent
3. **XGBoost**: Optimized gradient boosting with L1/L2 regularization
4. **Extra Trees (ET)**: Extremely randomized trees

**Hyperparameter Tuning:**
- **Method**: GridSearchCV with shuffled 5-fold cross-validation
- **Cross-validation**: `KFold(n_splits=5, shuffle=True, random_state=42)`
  - Ensures balanced representation of all genotypes in each fold
  - Prevents sequential bias from data ordering
- **Scoring metric**: R² (coefficient of determination)
- **Search space**: 
  - Random Forest: 32 combinations
  - Gradient Boosting: 64 combinations
  - XGBoost: 64 combinations
  - Extra Trees: 16 combinations

**Evaluation Metrics:**
- **Training R²**: Model fit on training data
- **Test R²**: Generalization performance (primary selection criterion)
- **RMSE**: Root mean squared error
- **CV R²**: 5-fold cross-validation R² (mean ± std)

### Step 5: Feature Importance Analysis

**Built-in Feature Importance (Gini/Gain Importance):**
- Measures total reduction in node impurity (variance for regression)
- Aggregated across all trees in the ensemble
- Higher values indicate features creating more homogeneous child nodes
- Normalized to sum to 1.0

**Output:**
- Ranked list of all 23 features
- Top 10 features displayed in console
- Top 15 features visualized in horizontal bar chart

### Step 6: Visualization

**Feature Importance Plot:**
- Horizontal bar chart of top 15 features
- Sorted by importance (highest at top)
- Saved to `output_plots/feature_importance.png` (300 DPI)

**Interaction Effects Plot:**
- Scatter plots of top 2 interaction features vs. log-transformed growth
- Color-coded by cross type (red=purebred, blue=hybrid)
- Shows differential G×E effects
- Saved to `output_plots/interaction_effects.png` (300 DPI)

---

## Expected Results

### Model Performance
Based on 360-day whole weight data:

**Model Ranking (by Test R²):**
1. **Extra Trees**: 0.4661 (Best)
2. **Random Forest**: 0.4660
3. **Gradient Boosting**: 0.4654
4. **XGBoost**: 0.4648

**Best Model Details (Extra Trees):**
- Test R²: 0.4661
- Test RMSE: 0.2133
- Train R²: 0.5405
- CV R²: 0.5270 ± 0.0340
- Optimal hyperparameters:
  - max_depth: 5
  - max_features: 'sqrt'
  - min_samples_leaf: 8
  - min_samples_split: 30
  - n_estimators: 400

### Top 10 Most Important Features (Extra Trees)

| Rank | Feature | Importance | Biological Interpretation |
|------|---------|------------|---------------------------|
| 1 | complexity_temperature | 0.0995 | Genetic complexity × temperature effect |
| 2 | is_hybrid | 0.0991 | Overall hybrid vigor effect |
| 3 | hybrid_temperature | 0.0947 | Hybrid status × temperature interaction |
| 4 | is_major_cross | 0.0805 | Major hybrid crosses indicator |
| 5 | genetic_complexity | 0.0758 | Genetic background (purebred vs hybrid) |
| 6 | hybrid_nutrition | 0.0616 | Hybrid status × nutrition interaction |
| 7 | geno_env_interaction | 0.0540 | Hybrid status × environment rank |
| 8 | complexity_env_rank | 0.0532 | Genetic complexity × environment interaction |
| 9 | complexity_nutrition | 0.0525 | Genetic complexity × nutrition interaction |
| 10 | complexity_temp_nutrition | 0.0452 | Three-way interaction effect |

**Key Findings:**
- **Balanced feature importance**: Top features show relatively even distribution (9-10% each)
- **Temperature interactions dominate**: Temperature-related features occupy top 3 positions
- **Hybrid status is crucial**: Direct hybrid effects and interactions are highly important
- **Complex interactions matter**: Two-way and three-way interactions show strong effects
- **No single dominant feature**: whole weight shows more distributed effects

### Model Selection Criteria
The best model (Extra Trees) is selected based on:
1. ✅ **Highest Test R²**: 0.4661 (best generalization)
2. ✅ **Moderate overfitting**: Train R² (0.5405) - Test R² (0.4661) = 0.0744
3. ✅ **Reasonable CV performance**: CV R² = 0.5270 ± 0.0340

---

## Reproducibility

### Random Seed Control
All stochastic processes use `RANDOM_SEED = 42`:
- Train-test split: `random_state=42`
- Model initialization: `random_state=42`
- Cross-validation folds: `KFold(shuffle=True, random_state=42)`
- Hyperparameter search: Deterministic given the above seeds

### Reproducibility Guarantee
Running the script multiple times with the same data will produce:
- ✅ Identical train-test splits
- ✅ Identical CV fold assignments
- ✅ Identical optimal hyperparameters
- ✅ Identical feature importance rankings
- ✅ Identical performance metrics (to 10 decimal places)

### Verification
To verify reproducibility, run:
```python
results1 = main(file_path)
results2 = main(file_path)

import numpy as np
np.allclose(
    results1['feature_importance']['Importance'].values,
    results2['feature_importance']['Importance'].values,
    rtol=1e-10
)  # Should return True
```

---

## Output Files

### Console Output
```
======================================================================
MACHINE LEARNING ANALYSIS FOR OYSTER G×E INTERACTIONS
======================================================================

[Step 1/6] Loading and configuring data...
Data loaded successfully. Shape: (94, 59)
Long-format transformation complete. Sample size: 3117

[Step 2/6] Engineering features...
Feature engineering complete: 12 base → 23 enhanced features

[Step 3/6] Preprocessing data...
Data split complete: 2493 train samples, 624 test samples

[Step 4/6] Training and comparing models...
======================================================================
MODEL TRAINING AND COMPARISON
======================================================================
Fitting 5 folds for each of 32 candidates, totalling 160 fits
Random Forest - Best parameters: {'ccp_alpha': 0.0, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 8, 'min_samples_split': 30, 'n_estimators': 400}
Fitting 5 folds for each of 64 candidates, totalling 320 fits
Gradient Boosting - Best parameters: {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 20, 'n_estimators': 200, 'subsample': 1.0}
Fitting 5 folds for each of 64 candidates, totalling 320 fits
XGBoost - Best parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 1.0}
Fitting 5 folds for each of 16 candidates, totalling 80 fits
Extra Trees - Best parameters: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 8, 'min_samples_split': 30, 'n_estimators': 400}

======================================================================
MODEL PERFORMANCE COMPARISON
======================================================================
            Model  Train_R2  Test_R2  Train_RMSE  Test_RMSE  CV_R2_Mean  CV_R2_Std
      Extra Trees    0.5405   0.4661      0.2172     0.2133      0.5270     0.0340
    Random Forest    0.5402   0.4660      0.2173     0.2133      0.5274     0.0342
Gradient Boosting    0.5406   0.4654      0.2172     0.2134      0.5268     0.0342
          XGBoost    0.5405   0.4648      0.2172     0.2135      0.5270     0.0344

Best model: Extra Trees
Test R²: 0.4661
Test RMSE: 0.2133

[Step 5/6] Analyzing feature importance...
======================================================================
FEATURE IMPORTANCE ANALYSIS
======================================================================
Top 10 Most Important Features (Built-in Importance):
                  Feature  Importance
   complexity_temperature      0.0995
                is_hybrid      0.0991
       hybrid_temperature      0.0947
           is_major_cross      0.0805
       genetic_complexity      0.0758
         hybrid_nutrition      0.0616
     geno_env_interaction      0.0540
      complexity_env_rank      0.0532
     complexity_nutrition      0.0525
complexity_temp_nutrition      0.0452

[Step 6/6] Generating visualizations...

======================================================================
ANALYSIS COMPLETE
======================================================================
Analysis results available in 'results' dictionary:
  - results['best_model_name']: Extra Trees
  - results['feature_importance']: DataFrame with 23 features
  - results['data_long']: DataFrame with 3117 samples
```

### Plot Files (if `plot_output_dir` specified)
- `feature_importance.png`: Horizontal bar chart (300 DPI)
- `interaction_effects.png`: Scatter plots (300 DPI)

---

## Customization

### Modify Hyperparameter Search Space

```python
# In train_best_xgb(), edit param_grid:
param_grid = {
    'n_estimators': [200, 300, 400],  # Add more values
    'learning_rate': [0.01, 0.05, 0.1],  # Finer grid
    'max_depth': [3, 4, 5],
    # ... add more parameters
}
```

### Add New Features

```python
# In enhance_features_with_nutrition():
df_long['new_feature'] = df_long['feature1'] * df_long['feature2']
enhanced_features.append('new_feature')
```

### Change Cross-Validation Folds

```python
# In all train_best_* functions:
cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)  # 10-fold
```

### Adjust Train-Test Split

```python
# In preprocess_data():
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,  # 70-30 split instead of 80-20
    random_state=RANDOM_SEED, 
    stratify=df_long['Environment']
)
```

---

## Troubleshooting

### Common Issues

**1. Different results between runs**
```
Problem: Feature importance changes each run
Solution: Ensure you're using the final version with KFold shuffle
Check: cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

**2. Excel file not found**
```
ValueError: Failed to load data: [Errno 2] No such file or directory
```
Solution: Verify file path, use raw string `r"C:/path/to/file.xlsx"`

**3. Column index error**
```
IndexError: Column index 60 exceeds data range
```
Solution: Ensure Excel has exactly 60 columns (12 genotypes × 5)

**4. Memory error**
```
MemoryError: Unable to allocate array
```
Solution: Reduce `n_estimators` or use fewer CV folds

**5. Slow execution**
```
Problem: GridSearchCV takes too long
```
Solution: 
- Reduce hyperparameter grid size
- Use `n_jobs=-1` for parallel processing (may affect reproducibility)
- Reduce `n_splits` from 5 to 3

---

## Citation

If you use this code in your research, please cite:


## Contact

**Primary Author:**  
Xianchao Bai  
Key Laboratory of Mariculture, Ministry of Education  
Ocean University of China, Qingdao 266003, China

**For questions or issues:**
- GitHub Issues: [Repository URL]/issues
- Email: [xcbai99@163.com]  
--- 

## Acknowledgments

This research was conducted at the Ocean University of China. We acknowledge:Shandong Province (2025LZGC036), Blue Seed Industry Innovation Project of Qingdao Institute of Blue Seed Industry (QDLYY-2024001), the Taishan Industrial Experts Program, and Guangdong Province (2024-MRB-00-001).
---

**Last Updated**: December 2025  
**Code Version**: 1.0  
**Python Version Tested**: 3.13
