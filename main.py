from preprocessor.volleyball_preprocessor import VolleyBallPreprocessor
from feature_engineering.feature_engineering import FeatureEngineeringPipeline
from sd.sd import SubgroupDiscovery
from sd.spurious import SwapRandomisationSignificance

# preprocess the project-specific data
preprocessor = VolleyBallPreprocessor(
    "sport_science_vb_data/ExerciseTrainingData.csv",
    "sport_science_vb_data/PlayerTrainingData.csv",
    "sport_science_vb_data/Wellness.csv",
    "sport_science_vb_data/Jumps.csv",
    "sport_science_vb_data/StrengthTraining.csv"
)
endurance_df, strength_df, wellness_df = preprocessor.load()

# apply feature engineering
fe = FeatureEngineeringPipeline()
daily_series_engineered_df = fe.transform(endurance_df, strength_df, wellness_df)

# run subgroup discovery on the constructed features
runner = SubgroupDiscovery(min_coverage=15, depth=1)
sd = runner.run(daily_series_engineered_df, target="wellness_total")

results = sd.asDataFrame().head(15)

print(results.columns.tolist())

print("Feature(s) and conditions ||| coverage ||| average ||| quality")
for _, r in results.iterrows():
    print(
        f"{r['Conditions']} ||| "
        f"{r['Coverage']} ||| "
        f"{r['Average']:.2f} ||| "
        f"{r['Quality']:.4f}"
    )
    
print("\nAverage of entire dataset:", daily_series_engineered_df["wellness_total"].mean())

# assess significance threshold of EV quality using swap randomisation
sig = SwapRandomisationSignificance(n_runs=10000, alpha=0.05)
threshold = sig.run(daily_series_engineered_df, "wellness_total")

print("\nSignificance threshold (p=0.05):", threshold)
