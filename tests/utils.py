import os   

def get_model_path(model, version):
    """
    Dynamically locates a model based on type and version.
    """
    base_dirs = [
        r"C:\Users\Myles\source\repos\savor_vision\models",
        r"E:\savor_vision\Savor_training"
    ]

    for base in base_dirs:
        potential_path = os.path.join(base, model, version, "weights", "best.pt")
        
        if os.path.exists(potential_path):
            return potential_path

    raise FileNotFoundError(f"ERROR: Could not find {model} ({version}) in known locations.")

