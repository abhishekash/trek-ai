from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import pipeline

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./trek_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    save_steps=100,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=10_000,
    logging_dir='./logs',
)

train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path='../trek-ai/trek_training_data/allcombined',
        block_size=24)

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

trainer.save_model("./trek_model")


prompt = 'Annapurna base camp '

generator = pipeline("text-generation", model=model, tokenizer="gpt2")
generated_text = generator(prompt, max_length=350, num_return_sequences=1)

print("\n\n")
print(prompt+ " : \n");
print("Generated text : "+generated_text[0]['generated_text'])