import json
import typer

app = typer.Typer()

@app.command()
def prepare(
    src_path: str = "training/llm/data/qa_raw.json",
    dst_path: str = "training/llm/data/rag_qa_dataset.jsonl"
):
    """
    Convert a JSON list of {"question", "context", "answer"} into Alpaca-style JSONL.
    """
    with open(src_path, encoding="utf-8") as f_src, open(dst_path, "w", encoding="utf-8") as f_dst:
        qa_list = json.load(f_src)
        for item in qa_list:
            record = {
                "instruction": item["question"],
                "input": item.get("context", ""),
                "output": item["answer"]
            }
            f_dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    typer.echo(f"✅ Prepared {dst_path}")

if __name__ == "__main__":
    app()
