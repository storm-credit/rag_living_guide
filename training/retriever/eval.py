import json
import typer
from sentence_transformers import SentenceTransformer, util
from training.retriever.dataset import TripletDataset

app = typer.Typer()

@app.command()
def evaluate(
    model_path: str = "trained_models/embedding_model",
    data_path: str = "training/retriever/data/structured_triplet_dataset.jsonl",
    k: int = 5
):
    """
    Basic IR evaluation: for each query, rank positives vs. negatives.
    Prints recall@k.
    """
    # 1) 모델 로드
    model = SentenceTransformer(model_path)

    # 2) 데이터 로드
    dataset = TripletDataset(data_path)
    total = len(dataset)
    hit = 0

    for q, pos, neg, _ in dataset:
        q_emb = model.encode(q, convert_to_tensor=True)
        pos_emb = model.encode(pos, convert_to_tensor=True)
        neg_emb = model.encode(neg, convert_to_tensor=True)

        # compute similarities
        sims = util.cos_sim(q_emb, [pos_emb, neg_emb])[0]
        # get top indices
        top_indices = sims.argsort(descending=True)[:k]
        # check if positive (index 0) is in top-k
        if 0 in top_indices:
            hit += 1

    recall = hit / total
    typer.echo(f"Recall@{k}: {recall:.4f} ({hit}/{total})")

if __name__ == "__main__":
    app()
