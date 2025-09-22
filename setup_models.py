import torch
import torchvision.models as models
from pathlib import Path


def setup_pretrained_models():
    """事前学習済みモデルをローカルに保存"""
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_configs = {
        'resnet18': {
            'model_func': models.resnet18,
            'weights': models.ResNet18_Weights.IMAGENET1K_V1
        },
        'resnet34': {
            'model_func': models.resnet34,
            'weights': models.ResNet34_Weights.IMAGENET1K_V1
        },
        'resnet50': {
            'model_func': models.resnet50,
            'weights': models.ResNet50_Weights.IMAGENET1K_V1
        }
    }
    
    for model_name, config in model_configs.items():
        model_path = models_dir / f"{model_name}_pretrained.pth"
        
        if model_path.exists():
            print(f"{model_name} already exists, skipping...")
            continue
        
        print(f"Downloading {model_name}...")
        model = config['model_func'](weights=config['weights'])
        
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model_name} to {model_path}")
    
    print("All models downloaded successfully!")
    
    total_size = sum(f.stat().st_size for f in models_dir.glob("*.pth"))
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")


def load_local_model(model_name: str):
    """ローカル保存されたモデルをロード"""
    models_dir = Path("models/pretrained")
    model_path = models_dir / f"{model_name}_pretrained.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"ローカルモデルが見つかりません: {model_path}")
    
    return torch.load(model_path, map_location='cpu')


if __name__ == "__main__":
    setup_pretrained_models()