# Example Data Documentation

## Overview

This file (`example-data.xlsx`) contains real experimental data from a 360-day field trial evaluating whole weight performance of allotetraploid and autotetraploidoysters under different environmental conditions.

**Data Type:** 360-day whole weight measurements (g)  
**Sample Size:** 3,117 individuals (after quality control)  
**Experimental Design:** 12 genotypes × 3 environments × ~89 individuals  
**Culture Period:** 360 days  
**Measurement:** Whole weight (shell + soft tissue, grams)

---

## Data Structure

### File Format
- **Format:** Excel (.xlsx)
- **Sheet Name:** "Sheet1"
- **Dimensions:** 94 rows × 59 columns
- **Header Row:** Row 2 (Python index 1)
- **Data Rows:** Rows 3-93 (Python indices 2-92)

### Column Organization

**Pattern:** Each genotype occupies 5 columns
```
[Index Column, Environment_1, Environment_2, Environment_3, Empty Column]
```

**Total:** 12 genotypes × 5 columns = 60 columns (column 0-59)

### Genotype Order (Left to Right)

| Position | Genotype | Shell Color | Cross Type | Description |
|----------|----------|-------------|------------|-------------|
| 1 | NGA | Normal | C.gigas♀ × C.angulata♂ | Forward hybrid |
| 2 | BGA | Black | C.gigas♀ × C.angulata♂ | Forward hybrid |
| 3 | GGA | Golden | C.gigas♀ × C.angulata♂ | Forward hybrid |
| 4 | NAG | Normal | C.angulata♀ × C.gigas♂ | Reverse hybrid |
| 5 | BAG | Black | C.angulata♀ × C.gigas♂ | Reverse hybrid |
| 6 | GAG | Golden | C.angulata♀ × C.gigas♂ | Reverse hybrid |
| 7 | NGG | Normal | C.gigas♀ × C.gigas♂ | C.gigas purebred |
| 8 | NAA | Normal | C.angulata♀ × C.angulata♂ | C.angulata purebred |
| 9 | BGG | Black | C.gigas♀ × C.gigas♂ | C.gigas purebred |
| 10 | BAA | Black | C.angulata♀ × C.angulata♂ | C.angulata purebred |
| 11 | GGG | Golden | C.gigas♀ × C.gigas♂ | C.gigas purebred |
| 12 | GAA | Golden | C.angulata♀ × C.angulata♂ | C.angulata purebred |

**Nomenclature:**
- **First letter**: Shell color (N=Normal, B=Black, G=Golden)
- **Last two letters**: Cross type (GG, GA, AG, AA)

### Environment Order (Per Genotype)

| Column Position | Environment | Location | Mean Temperature | Description |
|-----------------|-------------|----------|------------------|-------------|
| 1 (index) | - | - | - | Genotype identifier |
| 2 | rc | Rongcheng | 11.60°C | Coldest site |
| 3 | rs | Rushan | 12.72°C | Intermediate site |
| 4 | jn | Jiaonan | 13.64°C | Warmest site |
| 5 | - | - | - | Empty column |

---

## Data Values

### Measurement Units
- **Whole weight:** Grams (g)
- **Precision:** 2 decimal places
- **Range:** Approximately 16-34 g (varies by genotype and environment)

### Missing Values
- **Total cells:** 5,546 (94 rows × 59 columns)
- **Missing values:** ~2,232 (40.25%)
- **Reason:** data quality control, outlier removal

### Quality Control Applied
- Removed the data with extremely small individual measurement values
- Original sample sizes varied by genotype-environment combination

---

## Expected Analysis Results

When running `oyster_ml_analysis.py` on this data, you should obtain:

### Data Processing
```
Long-format transformation complete. Sample size: 3117
Data split complete: 2493 train samples, 624 test samples
```

### Best Model Performance
```
Best model: Extra Trees
Test R²: 0.4661
Test RMSE: 0.2133
CV R²: 0.5270 ± 0.0340
```

### Top 5 Features
1. complexity_temperature (0.0995)
2. is_hybrid (0.0991)
3. hybrid_temperature (0.0947)
4. is_major_cross (0.0805)
5. genetic_complexity (0.0758)

---

## Usage Example

### Basic Usage
```python
from oyster_ml_analysis import main

# Run analysis on example data
file_path = "example-data.xlsx"
results = main(file_path)

# Check results match expected values
print(f"Best Model: {results['best_model_name']}")  # Should be "Extra Trees"
print(f"Sample Size: {len(results['data_long'])}")  # Should be 3117
```

### Verify Reproducibility
```python
# Run twice to verify identical results
results1 = main(file_path)
results2 = main(file_path)

import numpy as np
importance_match = np.allclose(
    results1['feature_importance']['Importance'].values,
    results2['feature_importance']['Importance'].values,
    rtol=1e-10
)
print(f"Results identical: {importance_match}")  # Should be True
```

---

## Data Interpretation

### Biological Context

**Experimental Setup:**
- **Species:** Pacific oyster (*Crassostrea gigas*) and Portuguese oyster (*C. angulata*)
- **Ploidy:** Tetraploid (4n)
- **Cross Types:**
  - Autotetraploid: GG (gigas × gigas), AA (angulata × angulata)
  - Allotetraploid: GA (gigas♀ × angulata♂), AG (angulata♀ × gigas♂)
- **Shell Color Groups:** Normal (wild-type), Black (selected line), Golden (selected line)

**Environmental Gradient:**
- Temperature range: 11.60-13.64°C (2.04°C difference)
- Nutrition proxy: Positively correlated with temperature (priori assumption)
- Culture duration: 360 days (full grow-out cycle)

---

## Data Privacy and Ethics

### Data Sharing Statement
This example dataset is provided for:
- ✅ Reproducibility verification
- ✅ Code testing and validation
- ✅ Educational purposes
- ✅ Method development and comparison

### Usage Restrictions
- ❌ Do not use for commercial breeding without permission
- ❌ Do not redistribute without attribution
- ⚠️ Results derived from this data should cite the original publication

```

---

## Data Provenance

### Experimental Site
- **Location:** Rushan, Jiaonan, and Rongcheng, Shandong Province, China
- **Culture System:** lantern nets culture in coastal waters
- **Experimental Period:** [day90] to [day360]

### Measurements
- **Equipment:** Electronic balance (precision: 0.01g)
- **Protocol:** 
  1. Individual oysters removed from culture lantern nets
  2. Excess water drained for 1 minute
  3. Whole oyster weighed (shell + tissue)
  4. Recorded to 2 decimal places

---

## Troubleshooting

### Common Issues

**1. "Sample size doesn't match"**
```
Expected: 3117
Your result: Different number
```
**Possible causes:**
- Different outlier removal threshold
- Excel file corruption
- Different pandas/openpyxl versions

**Solution:** Re-download example-data.xlsx

**2. "Feature importance values differ"**
```
Expected Top feature: complexity_temperature (0.0995)
Your result: Different value
```
**Possible causes:**
- Not using the final code with KFold shuffle
- Different random seed
- Different scikit-learn version

**Solution:** Ensure using `oyster_ml_analysis_final.py` with `RANDOM_SEED=42`

**3. "Best model is different"**
```
Expected: Extra Trees
Your result: Different model
```
**This is NOT an error if:**
- Test R² values are very close (difference < 0.001)
- In such cases, model selection may flip due to floating-point precision

---

## Technical Notes

### File Size
- **Uncompressed:** 31 KB
- **Git-friendly:** Yes (small binary file)

### Compatibility
- **Excel versions:** 2007+ (.xlsx format)
- **Python libraries:** pandas + openpyxl
- **Tested on:** Windows

---

## Related Files

- `oyster_ml_analysis.py` - Main analysis script
- `README.md` - Main documentation
- `example-data.xlsx` - This example dataset
- `requirements.txt` - Python package dependencies

---

## Questions and Support

For questions about this example data:
- **GitHub Issues:** [Repository URL]/issues
- **Email:** [xcbai99@163.com]
- **Tag:** Use `[example-data]` in issue titles

---

**Last Updated:** December 2025  
**Data Version:** 1.0 (360-day whole weight)  
**File Format Version:** Excel 2007+ (.xlsx)
