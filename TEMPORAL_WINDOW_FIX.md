# Fix for Temporal Feature Windows

## Issue
Temporal feature windows were starting at the current day (t), instead of at t-1. This caused data leakage where features for day t would include data from day t itself.

## Problem Example
For a 7-day rolling mean on 2024-01-07:
- **Before fix (incorrect)**: Includes days [2024-01-01, ..., 2024-01-07]
- **After fix (correct)**: Includes days [2023-12-31, ..., 2024-01-06]

When predicting a target variable (like wellness) for day t, we should only use historical data up to t-1, not including t.

## Solution
In `feature_engineering/feature_engineering.py`, line 101:
- **Before**: `rolling = series.rolling(window=w)`
- **After**: `rolling = series.shift(1).rolling(window=w)`

The `.shift(1)` operation shifts the series forward by one position, effectively making the rolling window look back from t-1 instead of t.

## Technical Details
The fix ensures:
1. No data leakage: Features for day t don't include data from day t
2. Proper temporal ordering: Rolling windows use only historical data
3. Consistent with machine learning best practices: Features are computed using only information available before prediction time

## Impact
- All rolling aggregate features (mean, max, min, std) across all temporal windows (3, 5, 7, 14, 21, 28 days) are now correctly calculated
- ACWR (Acute/Chronic Workload Ratio) calculation automatically benefits from this fix since it uses the rolling features
- First w-1 days will have NaN values for w-day windows (expected behavior, as there's insufficient historical data)

## Testing
A test file `test_temporal_window_fix.py` has been created to verify the fix. The test:
1. Creates a simple dataset with known values
2. Applies the feature engineering pipeline
3. Verifies that rolling means exclude the current day
4. Confirms that calculations use only t-1 and earlier data

To run the test (requires pandas and numpy):
```bash
python3 test_temporal_window_fix.py
```
