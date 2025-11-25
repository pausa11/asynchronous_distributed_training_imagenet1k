import webdataset as wds
from PIL import Image
import io

def decode_cls(data):
    return data.decode("utf-8")

def inspect_raw():
    val_url = "https://storage.googleapis.com/caso-estudio-2/tiny-imagenet-wds/val/val-000000.tar"
    print(f"Inspecting raw validation data from: {val_url}")
    
    dataset = (
        wds.WebDataset(val_url)
        .decode(wds.handle_extension("cls", decode_cls), "pil")
        .to_tuple("jpg;png", "cls")
    )
    
    for i, (image, label) in enumerate(dataset):
        if i >= 5:
            break
        
        print(f"Sample {i}: Label: {label}, Size: {image.size}")

if __name__ == "__main__":
    inspect_raw()
