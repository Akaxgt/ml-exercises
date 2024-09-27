import joblib

def load_model(model_path:str):
    print(f"Loading model from {model_path}")

def write_model(model,model_path:str):
    print(f"Writing model to {model_path}")
    joblib.dump(model,model_path)

if __name__=="__main__":
    model_path="model.pkl"
    load_model(model_path)