import json

if __name__ == "__main__":
    for split in ["val"]:
        with open(
            f"/media/administrator/3227344f-bcd2-48f5-946e-78d2e7db3d26/vqa-frame/ori_clevr/CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json",
            "r",
        ) as infile:
            data = json.load(infile)

        new_data = {}
        new_data["info"] = data["info"]
        scenes = data["scenes"]
        scenes = [s for s in scenes if s["image_index"] <= 32]
        new_data["scenes"] = scenes

        with open(f"CLEVR_{split}_scenes.json", "w") as outfile:
            json.dump(new_data, outfile, indent=2)
