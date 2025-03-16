import requests


TOKEN = ...                         # Your token here
URL = "http://149.156.182.9:6060/task-5/submit"
agent_file = "./agent.py"

# These are just some random .pt files as an example
weights_file = './example_weights/example_weights.pt'
weights_file_2 = './example_weights/example_weights_2.pt'


def submitting_example():
    with open(agent_file, "rb") as agent, open(weights_file, "rb") as weight, open(weights_file_2, "rb") as weight_2:
        files = [
            ("agent_file", ("agent.py", agent, "application/octet-stream")),
            ("files", ("example_weights.pt", weight, "application/octet-stream")),
            ("files", ("example_weights_2.pt", weight_2, "application/octet-stream")),
            # ... You can add up to 5 files with weights here
        ]

        result = requests.post(
            URL,
            headers={"token": TOKEN},
            files=files
        )

        print(result.status_code, result.text)


if __name__ == '__main__':
    submitting_example()
