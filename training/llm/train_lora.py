import typer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, get_peft_config, TaskType
from datasets import load_dataset

app = typer.Typer()

@app.command()
def train(
    model_name: str = "beomi/KoAlpaca-Polyglot-5.8B",
    dataset_path: str = "training/llm/data/rag_qa_dataset.jsonl",
    output_dir: str = "trained_models/llm_lora",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4
):
    """
    Fine-tune LLM with LoRA on instruction-following QA data.
    """
    # 1) Load dataset
    ds = load_dataset("json", data_files=dataset_path, split="train")
    # 2) Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    # 3) Prepare PEFT config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, lora_alpha=16, lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    # 4) Tokenize
    def tokenize_fn(ex):
        prompt = f"""### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"""
        tokens = tokenizer(prompt, truncation=True, max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    # 5) Training args
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
        load_best_model_at_end=True
    )
    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized
    )
    trainer.train()
    model.save_pretrained(output_dir)
    typer.echo(f"✅ LLM LoRA fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    app()
