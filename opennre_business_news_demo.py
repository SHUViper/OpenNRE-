"""
OpenNRE practical scenario demo: relation extraction for business news.

This script shows how OpenNRE can be used after an upstream NER/entity-span
module has already found entity mentions. It is intentionally small and focused:
OpenNRE performs relation classification for a given sentence and entity pair.
"""

import opennre


def main():
    model = opennre.get_model("wiki80_bertentity_softmax")

    samples = [
        {
            "text": "Steve Jobs founded Apple in California.",
            "h": {"pos": (0, 10)},
            "t": {"pos": (19, 24)},
            "description": "person-company relation",
        },
        {
            "text": "Microsoft was founded by Bill Gates and Paul Allen.",
            "h": {"pos": (25, 35)},
            "t": {"pos": (0, 9)},
            "description": "founder-company relation",
        },
    ]

    for sample in samples:
        relation, score = model.infer(sample)
        print(sample["description"])
        print("text:", sample["text"])
        print("prediction:", relation, "score:", round(score, 4))
        print()


if __name__ == "__main__":
    main()
