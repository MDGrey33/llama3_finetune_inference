from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import os

# file_path = os.getenv('PROJECT_ROOT', '/llm_finetune/fintune_dataset.txt')
file_path = "/Users/roland/code/llama3_finetune_inference/llm_finetune/finetune_dataset.txt"
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'


def load_dataset(tokenizer, file_path=file_path, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM does not use masked language modeling
    )
    return dataset, data_collator


def main(model_name=model_name, file_path=file_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    try:
        with open(file_path, 'r') as f:
            print("Dataset loaded successfully, starting preprocessing...")
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure the file is in the correct directory.")
        return

    train_dataset, data_collator = load_dataset(tokenizer, file_path)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Adjusted for potential resource constraints
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to="all",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    # Start training
    print("Training is starting...")
    trainer.train()
    print("Training completed. Check './results' for output models and './logs' for logs.")

    # Save the model
    model_path = './trained_model'
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
