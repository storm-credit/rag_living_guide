import json
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Triplet format dataset for dense retriever training.
    Each item: (query, positive, negative, metadata)
    """
    def __init__(self, path: str):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "query": item["query"],
                    "positive": item["positive"],
                    "negative": item["negative"],
                    "meta": item.get("meta", {})
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["query"], s["positive"], s["negative"], s["meta"]
