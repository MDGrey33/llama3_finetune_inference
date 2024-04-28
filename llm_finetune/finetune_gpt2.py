import gpt_2_simple as gpt2
import os

# Download the model if it's not already present
model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)

# Prepare your dataset
file_name = "shakespeare.txt"

# Start a TensorFlow session and finetune the model
sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )

# Generate and print text from the finetuned model
generated_texts = gpt2.generate(sess, return_as_list=True)
for text in generated_texts:
    print(text)
