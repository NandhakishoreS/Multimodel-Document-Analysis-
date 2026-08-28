from datasets import load_dataset

print("Loading SROIE dataset...")

dataset = load_dataset(
    "jsdnrs/ICDAR2019-SROIE",
    split="all"
)

print("\nDataset loaded successfully!")
print(dataset)

print("\nNumber of samples:")
print(len(dataset))

print("\nDataset features:")
print(dataset.features)