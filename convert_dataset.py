import json
from tqdm import tqdm

json_file = "train.jsonl"
image_dir = "/datasets/LLaVA-CoT-100k/"
save_dir = ""

def main(json_file, image_dir, save_dir):
    new_dataset = []
    new_pipe_dataset = [] # For QwenVL pipeline

    with open(json_file, "r") as f:
        for line in tqdm(f, total=98582):
            data = json.loads(line)

            new_data = {"id": data["id"], "image": data["image"], "conversations": []}
            new_pipe_data = {"id": data["id"], "image": data["image"], "conversations": []}

            convs = data["conversations"]
            for i, conv in enumerate(convs):
                if conv["from"] == "human":
                    new_data["conversations"].append({"role": "user", "content": conv["value"]})
                    if i != 0:
                        new_pipe_data["conversations"].append({"role": "user", "content": conv["value"]})
                    else:
                        new_pipe_data["conversations"].append({"role": "user", 
                                                               "content": [
                                                                   {
                                                                       "type": "image",
                                                                       "image": image_dir + data["image"]
                                                                   },
                                                                   {
                                                                       "type": "text",
                                                                       "text": conv["value"]
                                                                   }
                                                               ]})
                elif conv["from"] == "gpt":
                    new_data["conversations"].append({"role": "assistant", "content": conv["value"]})
                    new_pipe_data["conversations"].append({"role": "assistant", "content": conv["value"]})
            new_dataset.append(new_data)
            new_pipe_dataset.append(new_pipe_data)

    with open(save_dir + "train_new.json", "w") as f:
        json.dump(new_dataset, f, indent=4)

    with open(save_dir + "train_pipe.json", "w") as f:
        json.dump(new_pipe_dataset, f, indent=4)


if __name__ == "__main__":
    main(json_file, image_dir, save_dir)
