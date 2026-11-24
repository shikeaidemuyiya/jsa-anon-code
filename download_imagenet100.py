"""Download ImageNet-100 (100-class ImageNet subset) from HuggingFace and save to disk."""
from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("clane9/imagenet-100")
    print(ds)
    ds.save_to_disk("./imagenet100")
    print("SAVED")
