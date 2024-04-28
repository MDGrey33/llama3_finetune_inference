import transformers
import os

# Get the project root directory from the environment variable
project_root = os.getenv('PROJECT_ROOT')

# Construct the model path using the project root
model_path = os.path.join(project_root, 'llm_finetune', 'trained_model_v1')

# Load the tokenizer and model from the fine-tuned checkpoint
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = transformers.AutoModelForCausalLM.from_pretrained(model_path)

# Initialize the pipeline with the fine-tuned model
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # assuming you are using GPU index 0, change if different
)

messages = [
    {"role": "system", "content": "You are a smart, and helpful AI assistant, you will do everything you can to help."},
    {"role": "user", "content": "Tell me what you know about Nur AI the software and Roland Abou Younes its creator, be precise and factual, only share what you are certain of."},
]


# Combine messages into a single string
prompt_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])

# Generate output using the pipeline
outputs = pipeline(
    prompt_text,
    max_length=500,  # Reduced max_length to limit verbosity and encourage precision
    truncation=True,
    num_return_sequences=1,
    temperature=0.2,  # Lower temperature to reduce randomness and increase predictability
    top_k=5,  # Limit the number of candidate tokens considered at each step
    top_p=0.1,  # Use nucleus sampling to further restrict token selection to the most likely tokens
    no_repeat_ngram_size=2  # Prevent repeating the same n-grams to avoid redundant content
)
print(outputs[0]['generated_text'])
