import json

if __name__ == "__main__":
    for split in ["test", "train", "val"]:
        with open(
            f"/media/administrator/3227344f-bcd2-48f5-946e-78d2e7db3d26/vqa-frame/ori_clevr/CLEVR_v1.0/questions/CLEVR_{split}_questions.json",
            "r",
        ) as infile:
            data = json.load(infile)

        new_data = {}
        new_data["info"] = data["info"]
        new_data["questions"] = data["questions"][: 10 + 30 * (split == "train")]

        with open(f"CLEVR_{split}_questions.json", "w") as outfile:
            json.dump(new_data, outfile, indent=2)
