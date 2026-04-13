from data_pipeline import create_stratified_splits

print("Scanning dataset and generating stratified splits...")

try:
    # This calls your function and generates the JSON files
    splits = create_stratified_splits()
    print("\nSPLITS SUCCESSFULLY GENERATED!")
    print("OSAMA load the data for training.")
except Exception as e:
    print(f"\nFAILED TO GENERATE SPLITS: {e}")