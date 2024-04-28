import os

print("""
This project intends to put out there an example of how to do the following inside your ide: 
1. load Llama3 and run inference on (poetry run python ./llm_inference/llama-3-8b-instruct.py)
2. fine tune llama3 model based on raw text file (poetry run python ./llm_finetune/finetune-llama3.py)
3. run inference on newly fintuned model with your data (poetry run python ./llm_inference/llama-nur-inference.py)

This project was run on a macbook pro M3 max with 128 GB ram overnight.

create a hugging face token and request permission tot he model you want to use
run pwd in the folder you cloned to get your project root

create a .env file and add the following to it
HF_TOKEN=yourhuggingfacetoken
PROJECT_ROOT=yourprojectroot
\n
""")