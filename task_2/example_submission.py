import base64
import io
import json
import numpy as np
import onnxruntime as ort
import pickle
import requests
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from typing import Tuple


TOKEN = ...                         # Your token here
SUBMIT_URL = "149.156.182.9:6060/task-2/submit"
RESET_URL = "149.156.182.9:6060/task-2/reset"
QUERY_URL = "149.156.182.9:6060/task-2/query"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


def generate_random_image():
    """Generates a random 32x32 RGB image and returns it as bytes."""
    random_pixels = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(random_pixels, 'RGB')

    # Save to a BytesIO buffer
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def quering_random():
    files = [("files", ("image2.png", generate_random_image(), "image/png")) for _ in range(1000)]
    response = requests.post(
        QUERY_URL,
        headers={"token": TOKEN},
        files=files
    )
    if response.status_code == 200:
        buffer = io.BytesIO(response.content)
        np_array = np.load(buffer)
        print(np_array.shape)
        print(np_array)

def reset_example():
    response = requests.post(
        RESET_URL,
        headers={"token": TOKEN}
    )

    print(response.text)

def quering_example():
    dataset = torch.load(...)                   # Path to ModelStealingPub.pt
    images = [dataset.imgs[idx] for idx in np.random.permutation(1000)]

    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        image_data.append(img_base64)

    payload = json.dumps(image_data)
    response = requests.post(QUERY_URL, headers={"token": TOKEN}, files={"file": payload})
    if response.status_code == 200:
        representation = response.json()["representations"]
    else:
        raise Exception(
            f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
        )
    # Store the output in a file.
    # Be careful to store all the outputs from the API since the number of queries is limited.
    with open('out.pickle', 'wb') as handle:
        pickle.dump(representation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Restore the output from the file.
    with open('out.pickle', 'rb') as handle:
        representation = pickle.load(handle)

    print(len(representation))


def submitting_example():

    # Create a dummy model
    model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 1024))

    path = 'dummy_submission.onnx'

    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        path,
        export_params=True,
        input_names=["x"],
    )

    # (these are being ran on the eval endpoint for every submission)
    with open(path, "rb") as f:
        model = f.read()
        try:
            stolen_model = ort.InferenceSession(model)
        except Exception as e:
            raise Exception(f"Invalid model, {e=}")
        try:
            out = stolen_model.run(
                None, {"x": np.random.randn(32, 3, 32, 32).astype(np.float32)}
            )[0][0]
        except Exception as e:
            raise Exception(f"Some issue with the input, {e=}")
        assert out.shape == (1024,), "Invalid output shape"

    response = requests.post(SUBMIT_URL, headers={"token": TOKEN}, files={"onnx_model": open(path, "rb")})
    print(response.status_code, response.text)



if __name__ == '__main__':
    quering_example()
    submitting_example()

