import kagglehub

path = kagglehub.dataset_download(
    "adamstanley537/pokemon-gen-1-image-dataset-27000-images"
)

print(f"Dataset znajduje się w: {path}")