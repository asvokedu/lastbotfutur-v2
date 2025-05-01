import os
import joblib

def convert_model_files(model_dir):
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            path = os.path.join(model_dir, filename)
            try:
                data = joblib.load(path)
                if isinstance(data, tuple):
                    if len(data) == 2:
                        model, features = data
                        converted = {"model": model, "features": features}
                        joblib.dump(converted, path)
                        print(f"{filename}: ✅ converted from tuple (2 items)")
                    elif len(data) >= 3:
                        model, features = data[0], data[1]
                        converted = {"model": model, "features": features}
                        joblib.dump(converted, path)
                        print(f"{filename}: ✅ converted from tuple (3+ items)")
                    else:
                        print(f"{filename}: ⚠️ unexpected tuple length ({len(data)})")
                elif isinstance(data, dict):
                    print(f"{filename}: ❌ already dict — skipped")
                else:
                    print(f"{filename}: ❌ unsupported type {type(data)} — skipped")
            except Exception as e:
                print(f"{filename}: ⚠️ error {e}")

if __name__ == "__main__":
    model_directory = "models"
    convert_model_files(model_directory)
