import typer
import os
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from training.retriever.dataset import TripletDataset

app = typer.Typer()

@app.command()
def train(
    data_path: str = "training/retriever/data/structured_triplet_dataset.jsonl",
    output_path: str = "trained_models/embedding_model",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5
):
    """
    Train a dense retriever using Triplet Loss (SBERT).
    """
    # 1) 데이터 로드
    dataset = TripletDataset(data_path)
    examples = [
        InputExample(texts=[q, p, n])
        for q, p, n, _ in dataset
    ]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    # 2) 모델 구성
    word_model = models.Transformer('sentence-transformers/all-mpnet-base-v2')
    pool = models.Pooling(word_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_model, pool])

    # 3) 학습 설정
    train_loss = losses.TripletLoss(model=model)
    typer.echo(f"Starting training: {len(dataset)} samples, epochs={epochs}")

    # 4) 학습 실행
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        output_path=output_path,
        optimizer_params={'lr': lr},
        show_progress_bar=True
    )
    typer.echo(f"✅ Retriever training complete, model saved to {output_path}")

if __name__ == "__main__":
    app()
# training/retriever/train.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, losses, models, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import json
from pathlib import Path

# ✅ Step 1. 데이터 로딩
def load_triplets(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(
                InputExample(texts=[item['query'], item['positive'], item['negative']])
            )
    return data

# ✅ Step 2. 모델 초기화 (SBERT 기반)
def build_model():
    word_embedding_model = models.Transformer('sentence-transformers/all-mpnet-base-v2')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

# ✅ Step 3. 학습 파라미터
DATA_PATH = 'retriever_training/data/structured_triplet_dataset.jsonl'
OUTPUT_PATH = 'trained_models/embedding_model/'
EPOCHS = 2
BATCH_SIZE = 16

if __name__ == "__main__":
    # 데이터 준비
    train_data = load_triplets(DATA_PATH)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    
    # 모델 구성
    model = build_model()
    
    # Triplet Loss
    train_loss = losses.TripletLoss(model=model)

    # 학습
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        output_path=OUTPUT_PATH,
        show_progress_bar=True
    )

    print(f"✅ 학습 완료. 모델 저장됨 → {OUTPUT_PATH}")
