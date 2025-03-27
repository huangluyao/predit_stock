import pickle


def save_obj(obj, file_path):
    buf = pickle.dumps(obj)
    with open(file_path, "wb") as f:
        f.write(buf)


def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.loads(f.read())
    return obj

