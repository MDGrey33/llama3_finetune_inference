## This project intends to share an example of how to do the following inpython: 
1. load Llama3 and run inference in python 
```poetry run python ./llm_inference/llama-3-8b-instruct.py```
2. fine tune llama3 model based on raw text file 
```poetry run python ./llm_finetune/finetune-llama3.py```
3. run inference on newly fintuned model with your data 
```poetry run python ./llm_inference/llama-nur-inference.py```
This project was run on a macbook pro M3 max with 128 GB ram overnight.
## Getting started
Clone repository
```git clone https://github.com/MDGrey33/llama3_finetune_inference.git```
cd to local folder
```cd llama3_finetune_inference```
setup poetry project 
```poetry install```
create a hugging face token and request permission to the model you want to use
run pwd in the folder you cloned to get your project rootcreate  .env file and add the following to it
```
HF_TOKEN=yourhuggingfacetoken
PROJECT_ROOT=yourprojectroot
```
1. load Llama3 and run inference on 
```poetry run python ./llm_inference/llama-3-8b-instruct.py```
2. fine tune llama3 model based on raw text file 
```poetry run python ./llm_finetune/finetune-llama3.py```
3. run inference on newly fintuned model with your data 
```poetry run python ./llm_inference/llama-nur-inference.py```
