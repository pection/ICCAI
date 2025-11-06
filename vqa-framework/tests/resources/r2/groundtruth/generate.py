import json
import os

if __name__ == "__main__":
    gt = {
        "test": {"p": {}, "q": {}, "s": {}, "f": {}},
        "val": {"p": {}, "q": {}, "s": {}, "f": {}},
        "train": {"p": {}, "q": {}, "s": {}, "f": {}},
    }

    for split in gt.keys():
        with open(
            os.path.join(
                "..",
                "ori_clevr",
                "CLEVR_v1.0",
                "questions",
                f"CLEVR_{split}_questions.json",
            ),
            "r",
        ) as infile:
            gt_data = json.load(infile)["questions"]

        for i, q in enumerate(gt_data):
            q["question_index"] = i  # Don't ask. Just don't ask.

        for q in gt_data:
            gt[split]["q"][q["question_index"]] = q["question"]
            if "program" in q:
                gt[split]["p"][q["question_index"]] = len(q["program"])
            else:
                gt[split]["p"][q["question_index"]] = None

            if "question_family_index" in q:
                gt[split]["f"][q["question_index"]] = q["question_family_index"]
            else:
                gt[split]["f"][q["question_index"]] = None

    with open("answer_key.json", "w") as outfile:
        json.dump(gt, outfile, indent=2)
