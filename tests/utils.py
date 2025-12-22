import os   

def find_data_path():
    possible_data_paths = [
        r"C:\Users\Myles\source\repos\savor_vision\dataset\data.yaml",
        r"E:\savor_vision\Savor_training\dataset\data.yaml"
    ]

    for path in possible_data_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Error: Data path not found in any known location.")



def find_model_path():
    possible_model_paths = [
        r"C:\Users\Myles\source\repos\savor_vision\Savor_training\mvp_run_22\weights\best.pt",
        r"E:\savor_vision\Savor_training\mvp_run_22\weights\best.pt"
    ]

    for path in possible_model_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Error: Model path not found in any known location.")