"""
Machine Learning Analysis for Tetraploid Oyster Genotype-by-Environment Interactions

This script implements ensemble machine learning models (Random Forest, Gradient Boosting,
XGBoost, Extra Trees) to evaluate G×E interactions in allotetraploid and autotetraploid oyster.

Author: Xianchao Bai
Institution: Key Laboratory of Mariculture, Ministry of Education, Ocean University of China, Qingdao 266003, China
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# ===========================================================================================
# GLOBAL CONFIGURATION
# ===========================================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Font settings for Chinese characters in plots (if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ===========================================================================================
# 1. DATA LOADING AND CONFIGURATION
# ===========================================================================================

def load_and_configure_data(file_path: str) -> tuple[pd.DataFrame, list[str], list[str], dict, dict]:
    """
    Load experimental data and configure genotypes and environments.

    Args:
        file_path: Path to the Excel data file

    Returns:
        df: Raw dataframe
        genotypes: List of 12 genotype combinations
        environments: List of 3 culture environments
        env_data: Dictionary of environmental parameters
        env_rank_mapping: Mapping of environments to ordinal ranks
    """
    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", header=1)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")

    # 12 genotype combinations: 3 color groups × 4 crossing types
    # N=Normal, B=Black, G=Golden shell color
    # GG=C.gigas×C.gigas, GA=C.gigas×C.angulata, AG=C.angulata×C.gigas, AA=C.angulata×C.angulata
    genotypes = ['NGA', 'BGA', 'GGA', 'NAG', 'BAG', 'GAG', 'NGG', 'NAA', 'BGG', 'BAA', 'GGG', 'GAA']

    # 3 culture environments with different temperature profiles
    environments = ['rc', 'rs', 'jn']

    # Environmental parameters (temperature in °C)
    env_data = {
        'rc': {'temperature': 11.60},
        'rs': {'temperature': 12.72},
        'jn': {'temperature': 13.64}
    }

    # Ordinal ranking of environments (1=lowest temperature, 3=highest)
    env_rank_mapping = {'rc': 1, 'rs': 2, 'jn': 3}

    return df, genotypes, environments, env_data, env_rank_mapping


def create_long_data(df: pd.DataFrame,
                     genotypes: list[str],
                     environments: list[str],
                     env_data: dict,
                     env_rank_mapping: dict) -> pd.DataFrame:
    """
    Transform wide-format data to long format with feature engineering.

    Data processing:
    1. Extract growth values for each genotype-environment combination
    2. Remove outliers using 3-sigma rule
    3. Apply log transformation: log(growth + 1)
    4. Create genetic and environmental features

    Args:
        df: Raw dataframe in wide format
        genotypes: List of genotype names
        environments: List of environment names
        env_data: Environmental parameters
        env_rank_mapping: Environment ordinal ranks

    Returns:
        df_long: Long-format dataframe with engineered features
    """
    data_long = []

    for genotype in genotypes:
        # Calculate column index for this genotype (5 columns per genotype)
        start_col = genotypes.index(genotype) * 5

        for env in environments:
            env_idx = environments.index(env)
            col_index = start_col + env_idx + 1

            if col_index >= df.shape[1]:
                raise IndexError(f"Column index {col_index} exceeds data range (max={df.shape[1] - 1})")

            # Extract growth values (rows 4-93, ~89 individuals)
            values = df.iloc[4:93, col_index]
            values_numeric = pd.to_numeric(values, errors='coerce')

            # Outlier removal: 3-sigma rule
            mean_val = values_numeric.mean()
            std_val = values_numeric.std()
            values_clean = values_numeric[
                (values_numeric >= mean_val - 3 * std_val) &
                (values_numeric <= mean_val + 3 * std_val)
                ].dropna()

            # Log transformation for normality
            values_log = np.log(values_clean + 1)

            # ===== Genetic Features =====
            # Binary indicator: is this a hybrid cross? (1=hybrid, 0=purebred)
            is_hybrid = 1 if genotype[1:] not in ['GG', 'AA'] else 0

            # Hybrid directionality: forward (GA) or reverse (AG)
            hybrid_forward = 1 if (genotype[1:] == 'GA' and genotype in ['BGA', 'GGA', 'NGA']) else 0
            hybrid_reverse = 1 if (genotype[1:] == 'AG' and genotype in ['BAG', 'GAG', 'NAG']) else 0

            # Genetic complexity: purebred=1, hybrid=2
            genetic_complexity = 1 if genotype[1:] in ['GG', 'AA'] else 2

            # Major cross indicator: primary hybrid crosses of interest
            is_major_cross = 1 if genotype in ['BGA', 'GGA', 'NGA', 'BAG', 'GAG', 'NAG'] else 0

            # Shell color groups (one-hot encoding)
            genetic_group_N = 1 if genotype[0] == 'N' else 0
            genetic_group_B = 1 if genotype[0] == 'B' else 0
            genetic_group_G = 1 if genotype[0] == 'G' else 0

            # ===== Environmental Features =====
            env_code = env_rank_mapping[env]

            # ===== Interaction Features =====
            # Genotype-environment interaction: hybrid status × environment rank
            geno_env_interaction = is_hybrid * env_code

            # Append each individual's data
            for val, val_log in zip(values_clean, values_log):
                data_long.append({
                    'Genotype': genotype,
                    'Environment': env,
                    'Growth': val,
                    'Growth_log': val_log,
                    'temperature': env_data[env]['temperature'],
                    'env_rank': env_code,
                    'is_hybrid': is_hybrid,
                    'hybrid_forward': hybrid_forward,
                    'hybrid_reverse': hybrid_reverse,
                    'genetic_complexity': genetic_complexity,
                    'is_major_cross': is_major_cross,
                    'genetic_group_N': genetic_group_N,
                    'genetic_group_B': genetic_group_B,
                    'genetic_group_G': genetic_group_G,
                    'geno_env_interaction': geno_env_interaction,
                })

    df_long = pd.DataFrame(data_long)
    print(f"Long-format transformation complete. Sample size: {len(df_long)}")
    return df_long


def enhance_features_with_nutrition(df_long: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Create enhanced features including nutrition rank and interaction terms.

    Feature engineering strategy:
    1. Nutrition rank as proxy for food availability (correlated with temperature)
    2. Polynomial features: temperature^2, log(temperature)
    3. Two-way interactions: genetic × environmental factors
    4. Three-way interactions: genetic × temperature × nutrition
    5. Composite environmental index

    Args:
        df_long: Long-format dataframe

    Returns:
        df_long: Dataframe with enhanced features
        feature_cols: List of all feature column names
        base_features: List of base feature names (before enhancement)
    """
    if df_long.empty:
        raise ValueError("Input dataframe is empty")

    # ===== Nutrition Rank =====
    # Ordinal ranking as proxy for food availability (1=low, 3=high)
    nutrition_rank_mapping = {'rc': 1, 'rs': 2, 'jn': 3}
    df_long['nutrition_rank'] = df_long['Environment'].map(nutrition_rank_mapping)

    # ===== Polynomial Features =====
    df_long['temperature_squared'] = df_long['temperature'] ** 2
    df_long['temperature_log'] = np.log(df_long['temperature'] + 1)

    # ===== Two-way Interaction Features =====
    # Genetic × Temperature
    df_long['hybrid_temperature'] = df_long['is_hybrid'] * df_long['temperature']
    df_long['complexity_temperature'] = df_long['genetic_complexity'] * df_long['temperature']

    # Genetic × Environment Rank
    df_long['complexity_env_rank'] = df_long['genetic_complexity'] * df_long['env_rank']

    # Nutrition × Temperature
    df_long['nutrition_temperature'] = df_long['nutrition_rank'] * df_long['temperature']

    # Genetic × Nutrition
    df_long['complexity_nutrition'] = df_long['genetic_complexity'] * df_long['nutrition_rank']
    df_long['hybrid_nutrition'] = df_long['is_hybrid'] * df_long['nutrition_rank']

    # ===== Three-way Interaction Features =====
    df_long['complexity_temp_nutrition'] = (
            df_long['genetic_complexity'] * df_long['temperature'] * df_long['nutrition_rank']
    )
    df_long['hybrid_temp_nutrition'] = (
            df_long['is_hybrid'] * df_long['temperature'] * df_long['nutrition_rank']
    )

    # ===== Composite Environmental Index =====
    # Weighted combination: (temperature + 2×nutrition) / 3
    df_long['environment_index'] = (df_long['temperature'] + df_long['nutrition_rank'] * 2) / 3

    # ===== Feature Lists =====
    base_features = [
        'temperature', 'env_rank', 'nutrition_rank', 'is_hybrid', 'hybrid_forward', 'hybrid_reverse',
        'genetic_complexity', 'is_major_cross', 'genetic_group_N', 'genetic_group_B', 'genetic_group_G',
        'geno_env_interaction'
    ]

    enhanced_features = [
        'temperature_squared', 'temperature_log',
        'hybrid_temperature', 'complexity_temperature', 'complexity_env_rank',
        'nutrition_temperature', 'complexity_nutrition', 'hybrid_nutrition',
        'complexity_temp_nutrition', 'hybrid_temp_nutrition', 'environment_index'
    ]

    # Validate all features exist
    missing_features = [f for f in (base_features + enhanced_features) if f not in df_long.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    feature_cols = base_features + enhanced_features

    print(f"Feature engineering complete: {len(base_features)} base → {len(feature_cols)} enhanced features")
    return df_long, feature_cols, base_features


# ===========================================================================================
# 2. DATA PREPROCESSING AND MODEL TRAINING
# ===========================================================================================

def preprocess_data(df_long: pd.DataFrame, feature_cols: list[str]) -> tuple:
    """
    Preprocess data: train-test split and feature scaling.

    Strategy:
    1. 80-20 train-test split with stratification by environment
    2. Standardize continuous features (mean=0, std=1) to improve model convergence
    3. Keep binary/categorical features unchanged

    Args:
        df_long: Long-format dataframe
        feature_cols: List of feature column names

    Returns:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors (log-transformed growth)
        scaler: Fitted StandardScaler object
    """
    X = df_long[feature_cols].copy()
    y = df_long['Growth_log'].copy()

    # Stratified split by environment to ensure balanced representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=df_long['Environment']
    )

    # Standardize continuous features
    continuous_features = ['temperature', 'env_rank', 'temperature_squared', 'temperature_log']
    scaler = StandardScaler()
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    print(f"Data split complete: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test, scaler


def train_best_rf(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Train Random Forest with hyperparameter tuning via GridSearchCV.

    Hyperparameter search space:
    - n_estimators: Number of trees [400, 500]
    - max_depth: Maximum tree depth [5, 6]
    - min_samples_split: Minimum samples to split node [30, 40]
    - min_samples_leaf: Minimum samples per leaf [8, 10]
    - max_features: Features per split ['sqrt']
    - ccp_alpha: Complexity parameter for pruning [0.0, 0.001]

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        best_rf: Best Random Forest model
        best_params: Best hyperparameters
    """
    param_grid = {
        'n_estimators': [400, 500],
        'max_depth': [5, 6],
        'min_samples_split': [30, 40],
        'min_samples_leaf': [8, 10],
        'max_features': ['sqrt'],
        'ccp_alpha': [0.0, 0.001]
    }

    rf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=1)

    # Use KFold with shuffle for proper cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='r2', n_jobs=1, verbose=1, refit=True
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"Random Forest - Best parameters: {grid_search.best_params_}")
    return best_rf, grid_search.best_params_


def train_best_gb(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Train Gradient Boosting with hyperparameter tuning.

    Hyperparameter search space:
    - n_estimators: Number of boosting stages [200, 300]
    - learning_rate: Shrinkage parameter [0.05, 0.1]
    - max_depth: Maximum tree depth [3, 4]
    - min_samples_split: Minimum samples to split [20, 30]
    - min_samples_leaf: Minimum samples per leaf [5, 10]
    - subsample: Fraction of samples per tree [0.8, 1.0]

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        best_gb: Best Gradient Boosting model
        best_params: Best hyperparameters
    """
    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'min_samples_split': [20, 30],
        'min_samples_leaf': [5, 10],
        'subsample': [0.8, 1.0]
    }

    gb = GradientBoostingRegressor(random_state=RANDOM_SEED)

    # Use KFold with shuffle for proper cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        gb, param_grid, cv=cv, scoring='r2', n_jobs=1, verbose=1, refit=True
    )
    grid_search.fit(X_train, y_train)

    best_gb = grid_search.best_estimator_
    print(f"Gradient Boosting - Best parameters: {grid_search.best_params_}")
    return best_gb, grid_search.best_params_


def train_best_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Train XGBoost with hyperparameter tuning.

    Hyperparameter search space:
    - n_estimators: Number of boosting rounds [200, 300]
    - learning_rate: Step size shrinkage [0.05, 0.1]
    - max_depth: Maximum tree depth [3, 5]
    - min_child_weight: Minimum sum of instance weight [1, 3]
    - subsample: Fraction of samples per tree [0.8, 1.0]
    - colsample_bytree: Fraction of features per tree [0.8, 1.0]

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        best_xgb: Best XGBoost model
        best_params: Best hyperparameters
    """
    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=1)

    # Use KFold with shuffle for proper cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=cv, scoring='r2', n_jobs=1, verbose=1, refit=True
    )
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    print(f"XGBoost - Best parameters: {grid_search.best_params_}")
    return best_xgb, grid_search.best_params_


def train_best_et(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Train Extra Trees with hyperparameter tuning.

    Extra Trees differs from Random Forest by using random thresholds for splits
    rather than optimal thresholds, which can reduce variance further.

    Hyperparameter search space:
    - n_estimators: Number of trees [400, 500]
    - max_depth: Maximum tree depth [5, 7]
    - min_samples_split: Minimum samples to split [30, 40]
    - min_samples_leaf: Minimum samples per leaf [8, 10]
    - max_features: Features per split ['sqrt']

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        best_et: Best Extra Trees model
        best_params: Best hyperparameters
    """
    param_grid = {
        'n_estimators': [400, 500],
        'max_depth': [5, 7],
        'min_samples_split': [30, 40],
        'min_samples_leaf': [8, 10],
        'max_features': ['sqrt']
    }

    et = ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=1)

    # Use KFold with shuffle for proper cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        et, param_grid, cv=cv, scoring='r2', n_jobs=1, verbose=1, refit=True
    )
    grid_search.fit(X_train, y_train)

    best_et = grid_search.best_estimator_
    print(f"Extra Trees - Best parameters: {grid_search.best_params_}")
    return best_et, grid_search.best_params_


# ===========================================================================================
# 3. MODEL COMPARISON AND EVALUATION
# ===========================================================================================

def compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> tuple:
    """
    Compare performance of four ensemble models and select the best one.

    Models evaluated:
    1. Random Forest (RF): Bootstrap aggregation of decision trees
    2. Gradient Boosting (GB): Sequential boosting with gradient descent
    3. XGBoost: Optimized gradient boosting with regularization
    4. Extra Trees (ET): Randomized decision trees with random thresholds

    Evaluation metrics:
    - R² (coefficient of determination): Proportion of variance explained
    - RMSE (root mean squared error): Prediction error magnitude
    - 5-fold cross-validation R²: Generalization performance

    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors

    Returns:
        results_df: DataFrame with model performance metrics
        best_model: Best performing model object
        best_model_name: Name of the best model
    """
    print("\n" + "=" * 70)
    print("MODEL TRAINING AND COMPARISON")
    print("=" * 70)

    # Train all models
    best_rf, _ = train_best_rf(X_train, y_train)
    best_gb, _ = train_best_gb(X_train, y_train)
    best_xgb, _ = train_best_xgb(X_train, y_train)
    best_et, _ = train_best_et(X_train, y_train)

    # Evaluate all models
    models = {
        'Random Forest': best_rf,
        'Gradient Boosting': best_gb,
        'XGBoost': best_xgb,
        'Extra Trees': best_et
    }

    # Use KFold for cross-validation in model comparison
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = []
    for name, model in models.items():
        # Training set performance
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Test set performance
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Cross-validation performance
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=1)
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()

        results.append({
            'Model': name,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'CV_R2_Mean': cv_r2_mean,
            'CV_R2_Std': cv_r2_std
        })

    results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(results_df.round(4).to_string(index=False))

    # Select best model based on test R²
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
    print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")

    return results_df, best_model, best_model_name


# ===========================================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ===========================================================================================

def analyze_feature_importance(best_model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_test: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    """
    Extract and analyze built-in feature importance from the best model.

    Built-in feature importance (Gini importance for tree models):
    - Measures the total reduction in node impurity (variance for regression)
    - Aggregated across all trees in the ensemble
    - Higher values indicate features that create more homogeneous child nodes

    Note: This is the only feature importance method used in this analysis.
    For more robust importance estimation, consider permutation importance or SHAP values,
    but these are not included here for simplicity.

    Args:
        best_model: Trained best model
        X_train: Training features
        X_test: Test features
        y_test: Test target
        feature_cols: List of feature names

    Returns:
        builtin_importance: DataFrame with features ranked by importance
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Extract built-in feature importance
    if hasattr(best_model, 'feature_importances_'):
        builtin_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features (Built-in Importance):")
        print(builtin_importance.head(10).round(4).to_string(index=False))

    else:
        # Fallback if model doesn't have feature_importances_ attribute
        builtin_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': 0
        })
        print("\nWarning: Model does not support built-in feature importance")

    return builtin_importance


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                            output_path: str = None):
    """
    Visualize top N most important features.

    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns
        top_n: Number of top features to display
        output_path: Optional path to save the figure
    """
    top_features = importance_df.head(top_n).copy()

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {output_path}")

    plt.show()


def plot_interaction_effects(df_long: pd.DataFrame, top_interaction_features: list[str],
                             output_path: str = None):
    """
    Visualize effects of top interaction features on growth.

    Creates scatter plots showing the relationship between interaction features and
    log-transformed growth, stratified by hybrid status.

    Args:
        df_long: Long-format dataframe with all data
        top_interaction_features: List of interaction feature names (containing '_')
        output_path: Optional path to save the figure
    """
    if not top_interaction_features:
        print("No interaction features to plot")
        return

    # Select top 2 interaction features for visualization
    n_features = min(2, len(top_interaction_features))

    plt.figure(figsize=(15, 6))
    for i, feature in enumerate(top_interaction_features[:n_features], 1):
        plt.subplot(1, n_features, i)

        # Scatter plot stratified by hybrid status
        sns.scatterplot(
            x=feature, y='Growth_log', hue='is_hybrid', data=df_long,
            alpha=0.6, palette=['red', 'blue'], hue_order=[0, 1]
        )

        plt.legend(title='Cross Type', labels=['Purebred (0)', 'Hybrid (1)'])
        plt.title(f'{feature} vs Growth (by Cross Type)', fontweight='bold')
        plt.xlabel(feature, fontsize=11)
        plt.ylabel('Log-transformed Growth', fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nInteraction effects plot saved to: {output_path}")

    plt.show()


# ===========================================================================================
# 5. MAIN WORKFLOW
# ===========================================================================================

def main(file_path: str, plot_output_dir: str = None):
    """
    Main workflow integrating all analysis steps.

    Workflow:
    1. Load and transform data to long format
    2. Engineer features (base + enhanced + interactions)
    3. Preprocess data (train-test split + scaling)
    4. Train and compare 4 ensemble models
    5. Analyze feature importance from best model
    6. Visualize top features and interaction effects

    Args:
        file_path: Path to Excel data file
        plot_output_dir: Optional directory to save plots
    """
    print("\n" + "=" * 70)
    print("MACHINE LEARNING ANALYSIS FOR OYSTER G×E INTERACTIONS")
    print("=" * 70)

    # Step 1: Data loading and configuration
    print("\n[Step 1/6] Loading and configuring data...")
    df, genotypes, environments, env_data, env_rank_mapping = load_and_configure_data(file_path)
    df_long = create_long_data(df, genotypes, environments, env_data, env_rank_mapping)

    # Step 2: Feature engineering
    print("\n[Step 2/6] Engineering features...")
    df_long, feature_cols, base_features = enhance_features_with_nutrition(df_long)

    # Step 3: Data preprocessing
    print("\n[Step 3/6] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df_long, feature_cols)

    # Step 4: Model training and comparison
    print("\n[Step 4/6] Training and comparing models...")
    results_df, best_model, best_model_name = compare_models(X_train, X_test, y_train, y_test)

    # Step 5: Feature importance analysis
    print("\n[Step 5/6] Analyzing feature importance...")
    builtin_importance = analyze_feature_importance(
        best_model, X_train, X_test, y_test, feature_cols
    )

    # Step 6: Visualization
    print("\n[Step 6/6] Generating visualizations...")

    # Plot feature importance
    importance_plot_path = f"{plot_output_dir}/feature_importance.png" if plot_output_dir else None
    plot_feature_importance(builtin_importance, top_n=15, output_path=importance_plot_path)

    # Plot interaction effects
    top_interaction_features = builtin_importance[
        builtin_importance['Feature'].str.contains('_')
    ]['Feature'].tolist()

    if top_interaction_features:
        interaction_plot_path = f"{plot_output_dir}/interaction_effects.png" if plot_output_dir else None
        plot_interaction_effects(df_long, top_interaction_features, output_path=interaction_plot_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Return key results for further use
    return {
        'results_df': results_df,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'feature_importance': builtin_importance,
        'data_long': df_long,
        'feature_cols': feature_cols
    }


# ===========================================================================================
# SCRIPT EXECUTION
# ===========================================================================================

if __name__ == "__main__":
    # Example usage:
    # Replace with your actual data file path
    file_path = r"C:/Users/Admin/Desktop/example-data.xlsx"

    # Optional: specify output directory for plots
    plot_output_dir = None  # Set to a directory path to save plots

    # Run main analysis
    results = main(file_path, plot_output_dir)

    # Access results
    print("\nAnalysis results available in 'results' dictionary:")
    print(f"  - results['best_model_name']: {results['best_model_name']}")
    print(f"  - results['feature_importance']: DataFrame with {len(results['feature_importance'])} features")
    print(f"  - results['data_long']: DataFrame with {len(results['data_long'])} samples")