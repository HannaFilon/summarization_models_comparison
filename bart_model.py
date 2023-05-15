from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AdamW
import datasets
from tqdm.auto import tqdm

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
if torch.cuda.is_available():
  model = model.to("cuda")

train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[0:1%]")
validation_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[0:1%]")

article_length=512
summary_length=64

def preprocess_data(batch):
  input_data = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=article_length)
  output_data = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=summary_length)

  batch["input_ids"] = input_data.input_ids
  batch["attention_mask"] = input_data.attention_mask
  batch["decoder_input_ids"] = output_data.input_ids
  batch["decoder_attention_mask"] = output_data.attention_mask
  batch["labels"] = output_data.input_ids.copy()
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

train_data = train_data.map(preprocess_data, batched=True,remove_columns=["article", "highlights", "id"])
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                           "decoder_attention_mask", "labels"])

validation_data = validation_data.map(preprocess_data, batched=True, remove_columns=["article", "highlights", "id"])
validation_data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                                                  "decoder_attention_mask", "labels"])

batch_size= 4

train_data = DataLoader(train_data, batch_size=batch_size)
validation_data = DataLoader(validation_data, batch_size=batch_size)

loss_function = CrossEntropyLoss()
optimizer_adam = AdamW(model.parameters(), lr=5e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_data)
num_validation_steps = num_epochs * len(validation_data)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer_adam,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

the_encoder = model.get_encoder()
the_decoder = model.get_decoder()
last_linear_layer = model.lm_head

progress_bar = tqdm(range(num_training_steps + num_validation_steps))

print("Training...")
for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    validation_loss = 0.0
    for batch in train_data:
        if torch.cuda.is_available():
            batch = {k: v.to('cuda') for k, v in batch.items()}

        encoder_output = the_encoder(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'])

        decoder_output = the_decoder(input_ids=batch['decoder_input_ids'],
                                     attention_mask=batch['decoder_attention_mask'],
                                     encoder_hidden_states=encoder_output[0],
                                     encoder_attention_mask=batch['attention_mask'])

        decoder_output = decoder_output.last_hidden_state
        lm_head_output = last_linear_layer(decoder_output)

        loss = loss_function(lm_head_output.view(-1, model.config.vocab_size),
                        batch['labels'].view(-1))
        training_loss += loss.item()

        loss.backward()
        optimizer_adam.step()
        lr_scheduler.step()
        optimizer_adam.zero_grad()
        progress_bar.update(1)

    model.eval()
    print("Validating...")
    for batch in validation_data:
        if torch.cuda.is_available():
            batch = {k: v.to('cuda') for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        validation_loss += loss
        progress_bar.update(1)

    training_loss = training_loss / len(train_data)
    validation_loss = validation_loss / len(validation_data)
    print(
        "Epoch {}:\tTraining Loss {:.2f}\t/\tValidation Loss {:.2f}".format(epoch + 1, training_loss, validation_loss))

output_dir = "/models/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
