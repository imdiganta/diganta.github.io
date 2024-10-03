import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, txt_file, csv_file):
        self.txt_data = open(txt_file, 'r').readlines()
        self.csv_data = pd.read_csv(csv_file)
        
        # Create a vocabulary mapping
        self.vocab = self.build_vocab(self.txt_data)
        
        # Debugging: print the vocabulary size
        print(f"Vocabulary Size: {len(self.vocab)}")

    def build_vocab(self, texts):
        # Create a vocabulary mapping from word to index
        vocab = {}
        for text in texts:
            for word in text.strip().split():
                if word not in vocab:
                    vocab[word] = len(vocab) + 1  # Start indexing from 1
        return vocab

    def __len__(self):
        return len(self.txt_data)

    def __getitem__(self, idx):
        text = self.txt_data[idx].strip()
        
        # Tokenize the text
        tokenized_text = torch.tensor([self.vocab[word] for word in text.split() if word in self.vocab], dtype=torch.long)
        
        return tokenized_text  # Return only the tokenized input

# Custom collate function for padding
def collate_fn(batch):
    # Pad sequences to the same length
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)  # Using 0 as padding index
    return padded_batch

# Transformer Model Class
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output)

# Training Function
def train(model, dataset, epochs, learning_rate):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            src = batch.to(device)
            tgt = batch.to(device)

            optimizer.zero_grad()
            output = model(src, tgt)
            loss = loss_fn(output.view(-1, output.size(-1)), tgt.view(-1))

            # Check for NaN values in the loss
            if torch.isnan(loss):
                print("NaN loss encountered! Skipping this batch.")
                continue

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Text Generation Function
def generate_text(model, input_text, max_length=20):
    model.eval()
    input_tensor = torch.tensor([dataset.vocab[word] for word in input_text.split() if word in dataset.vocab], dtype=torch.long).unsqueeze(0).to(device)

    generated = input_text

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, input_tensor)
            next_word_idx = output.argmax(-1)[-1, -1].item()  # Get the most probable next word

            # Check if next_word_idx is valid
            if next_word_idx == 0:  # 0 is the padding index
                break
            
            generated += ' ' + list(dataset.vocab.keys())[next_word_idx - 1]  # Append the predicted word
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_word_idx]]).to(device)), dim=1)

    return generated

# Main Execution
if __name__ == "__main__":
    # Load the dataset
    dataset = TextDataset('data.txt', 'data.csv')

    # Define model parameters
    vocab_size = len(dataset.vocab) + 1  # Include a padding index
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # Create model instance and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

    # Train the model
    train(model, dataset, epochs=10, learning_rate=0.001)

    # Generate text
    output = generate_text(model, "Hello", max_length=20)
    print("Generated Text:", output)



























































































































































































































# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# from torch.nn.utils.rnn import pad_sequence

# # Custom Dataset Class
# class TextDataset(Dataset):
#     def __init__(self, txt_file, csv_file):
#         self.txt_data = open(txt_file, 'r').readlines()
#         self.csv_data = pd.read_csv(csv_file)
        
#         # Create a vocabulary mapping
#         self.vocab = self.build_vocab(self.txt_data)
        
#         # Debugging: print the vocabulary size
#         print(f"Vocabulary Size: {len(self.vocab)}")

#     def build_vocab(self, texts):
#         # Create a vocabulary mapping from word to index
#         vocab = {}
#         for text in texts:
#             for word in text.strip().split():
#                 if word not in vocab:
#                     vocab[word] = len(vocab) + 1  # Start indexing from 1
#         return vocab

#     def __len__(self):
#         return len(self.txt_data)

#     def __getitem__(self, idx):
#         text = self.txt_data[idx].strip()
        
#         # Tokenize the text
#         tokenized_text = torch.tensor([self.vocab[word] for word in text.split() if word in self.vocab], dtype=torch.long)
        
#         return tokenized_text  # Return only the tokenized input

# # Custom collate function for padding
# def collate_fn(batch):
#     # Pad sequences to the same length
#     padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)  # Using 0 as padding index
#     return padded_batch

# # Transformer Model Class
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, src, tgt):
#         src = self.embedding(src)
#         tgt = self.embedding(tgt)
#         output = self.transformer(src, tgt)
#         return self.fc_out(output)

# # Training Function
# def train(model, dataset, epochs, learning_rate):
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             # Move the batch to the appropriate device
#             src = batch.to(device)
#             tgt = batch.to(device)

#             optimizer.zero_grad()
#             output = model(src, tgt)
#             loss = loss_fn(output.view(-1, output.size(-1)), tgt.view(-1))
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# # Text Generation Function
# def generate_text(model, input_text, max_length=50):
#     model.eval()
#     input_tensor = torch.tensor([dataset.vocab[word] for word in input_text.split() if word in dataset.vocab], dtype=torch.long).unsqueeze(0).to(device)  # Convert input to tensor

#     generated = input_text

#     for _ in range(max_length):
#         output = model(input_tensor, input_tensor)
#         next_word = output.argmax(-1)[-1].item()  # Get the most probable next word
#         generated += ' ' + list(dataset.vocab.keys())[next_word - 1]  # Append the predicted word
#         input_tensor = torch.cat((input_tensor, torch.tensor([[next_word]]).to(device)), dim=1)  # Update input

#     return generated

# # Main Execution
# if __name__ == "__main__":
#     # Load the dataset
#     dataset = TextDataset('data.txt', 'data.csv')

#     # Define model parameters
#     vocab_size = len(dataset.vocab) + 1  # Include a padding index
#     d_model = 512
#     nhead = 8
#     num_encoder_layers = 6
#     num_decoder_layers = 6

#     # Create model instance and move to device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

#     # Train the model
#     train(model, dataset, epochs=10, learning_rate=0.001)

#     # Generate text
#     output = generate_text(model, "Hello", max_length=20)
#     print("Generated Text:", output)
























































































































































































# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# from torch.nn.utils.rnn import pad_sequence

# # Custom Dataset Class
# class TextDataset(Dataset):
#     def __init__(self, txt_file, csv_file):
#         self.txt_data = open(txt_file, 'r').readlines()
#         self.csv_data = pd.read_csv(csv_file)
        
#         # Create a vocabulary mapping
#         self.vocab = self.build_vocab(self.txt_data)
        
#         # Debugging: print the vocabulary size
#         print(f"Vocabulary Size: {len(self.vocab)}")

#     def build_vocab(self, texts):
#         # Create a vocabulary mapping from word to index
#         vocab = {}
#         for text in texts:
#             for word in text.strip().split():
#                 if word not in vocab:
#                     vocab[word] = len(vocab) + 1  # Start indexing from 1
#         return vocab

#     def __len__(self):
#         return len(self.txt_data)

#     def __getitem__(self, idx):
#         text = self.txt_data[idx].strip()
        
#         # Tokenize the text
#         tokenized_text = torch.tensor([self.vocab[word] for word in text.split() if word in self.vocab], dtype=torch.long)
        
#         # Accessing the additional information
#         additional_info = self.csv_data.iloc[idx][self.csv_data.columns[1].strip()]
        
#         return tokenized_text  # Return only the tokenized input

# # Custom collate function for padding
# def collate_fn(batch):
#     # Pad sequences to the same length
#     padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)  # Using 0 as padding index
#     return padded_batch

# # Transformer Model Class
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, src, tgt):
#         src = self.embedding(src)
#         tgt = self.embedding(tgt)
#         output = self.transformer(src, tgt)
#         return self.fc_out(output)

# # Training Function
# def train(model, dataset, epochs, learning_rate):
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             # Here we will use the same batch for src and tgt for simplicity
#             src = batch.to(device)
#             tgt = batch.to(device)

#             optimizer.zero_grad()
#             output = model(src, tgt)
#             loss = loss_fn(output.view(-1, output.size(-1)), tgt.view(-1))
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# # Text Generation Function
# def generate_text(model, input_text, max_length=50):
#     model.eval()
#     input_tensor = torch.tensor([dataset.vocab[word] for word in input_text.split() if word in dataset.vocab], dtype=torch.long).unsqueeze(0).to(device)  # Convert input to tensor

#     generated = input_text

#     for _ in range(max_length):
#         output = model(input_tensor, input_tensor)
#         next_word = output.argmax(-1)[-1].item()  # Get the most probable next word
#         generated += ' ' + next_word  # Append the predicted word
#         input_tensor = torch.cat((input_tensor, torch.tensor([[next_word]]).to(device)), dim=1)  # Update input

#     return generated

# # Main Execution
# if __name__ == "__main__":
#     # Load the dataset
#     dataset = TextDataset('data.txt', 'data.csv')

#     # Define model parameters
#     vocab_size = len(dataset.vocab) + 1  # Include a padding index
#     d_model = 512
#     nhead = 8
#     num_encoder_layers = 6
#     num_decoder_layers = 6

#     # Create model instance
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

#     # Train the model
#     train(model, dataset, epochs=10, learning_rate=0.001)

#     # Generate text
#     output = generate_text(model, "Hello", max_length=20)
#     print("Generated Text:", output)










































































































































































# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim

# # Custom Dataset Class
# class TextDataset(Dataset):
#     def __init__(self, txt_file, csv_file):
#         self.txt_data = open(txt_file, 'r').readlines()
#         self.csv_data = pd.read_csv(csv_file)
        
#         # Create a vocabulary mapping
#         self.vocab = self.build_vocab(self.txt_data)
        
#         # Debugging: print the vocabulary size
#         print(f"Vocabulary Size: {len(self.vocab)}")

#     def build_vocab(self, texts):
#         # Create a vocabulary mapping from word to index
#         vocab = {}
#         for text in texts:
#             for word in text.strip().split():
#                 if word not in vocab:
#                     vocab[word] = len(vocab) + 1  # Start indexing from 1
#         return vocab

#     def __len__(self):
#         return len(self.txt_data)

#     def __getitem__(self, idx):
#         text = self.txt_data[idx].strip()
        
#         # Tokenize the text
#         tokenized_text = torch.tensor([self.vocab[word] for word in text.split() if word in self.vocab], dtype=torch.long)
        
#         # Accessing the additional information
#         additional_info = self.csv_data.iloc[idx][self.csv_data.columns[1].strip()]
        
#         return tokenized_text, tokenized_text  # Returning tokenized input as both src and tgt for simplicity

# # Transformer Model Class
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, src, tgt):
#         src = self.embedding(src)
#         tgt = self.embedding(tgt)
#         output = self.transformer(src, tgt)
#         return self.fc_out(output)

# # Training Function
# def train(model, dataset, epochs, learning_rate):
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             src, tgt = batch  # src and tgt are now tensors
#             src, tgt = src.to(device), tgt.to(device)

#             optimizer.zero_grad()
#             output = model(src, tgt)
#             loss = loss_fn(output.view(-1, output.size(-1)), tgt.view(-1))
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# # Text Generation Function
# def generate_text(model, input_text, max_length=50):
#     model.eval()
#     input_tensor = torch.tensor([dataset.vocab[word] for word in input_text.split() if word in dataset.vocab], dtype=torch.long).unsqueeze(0).to(device)  # Convert input to tensor

#     generated = input_text

#     for _ in range(max_length):
#         output = model(input_tensor, input_tensor)
#         next_word = output.argmax(-1)[-1].item()  # Get the most probable next word
#         generated += ' ' + next_word  # Append the predicted word
#         input_tensor = torch.cat((input_tensor, torch.tensor([[next_word]]).to(device)), dim=1)  # Update input

#     return generated

# # Main Execution
# if __name__ == "__main__":
#     # Load the dataset
#     dataset = TextDataset('data.txt', 'data.csv')

#     # Define model parameters
#     vocab_size = len(dataset.vocab) + 1  # Include a padding index
#     d_model = 512
#     nhead = 8
#     num_encoder_layers = 6
#     num_decoder_layers = 6

#     # Create model instance
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

#     # Train the model
#     train(model, dataset, epochs=10, learning_rate=0.001)

#     # Generate text
#     output = generate_text(model, "Hello", max_length=20)
#     print("Generated Text:", output)







































































































































# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim

# # Custom Dataset Class
# class TextDataset(Dataset):
#     def __init__(self, txt_file, csv_file):
#         self.txt_data = open(txt_file, 'r').readlines()
#         self.csv_data = pd.read_csv(csv_file)
        
#         # Print the DataFrame to check its structure
#         print("CSV DataFrame:")
#         print(self.csv_data.head())  # Debugging: print first few rows

#     def __len__(self):
#         return len(self.txt_data)

#     def __getitem__(self, idx):
#         text = self.txt_data[idx].strip()
        
#         # Ensure we're accessing the correct column
#         additional_info = self.csv_data.iloc[idx][self.csv_data.columns[1].strip()]
        
#         return text, additional_info

# # Transformer Model Class
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, src, tgt):
#         src = self.embedding(src)
#         tgt = self.embedding(tgt)
#         output = self.transformer(src, tgt)
#         return self.fc_out(output)

# # Training Function
# def train(model, dataset, epochs, learning_rate):
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             src, tgt = batch
#             src, tgt = src.to(device), tgt.to(device)

#             optimizer.zero_grad()
#             output = model(src, tgt)
#             loss = loss_fn(output.view(-1, output.size(-1)), tgt.view(-1))
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# # Text Generation Function
# def generate_text(model, input_text, max_length=50):
#     model.eval()
#     generated = input_text
#     input_tensor = torch.tensor([input_text]).unsqueeze(1).to(device)  # Convert input to tensor

#     for _ in range(max_length):
#         output = model(input_tensor, input_tensor)
#         next_word = output.argmax(-1)[-1].item()  # Get the most probable next word
#         generated += ' ' + str(next_word)  # Append the predicted word
#         input_tensor = torch.cat((input_tensor, torch.tensor([[next_word]]).to(device)), dim=0)  # Update input

#     return generated

# # Main Execution
# if __name__ == "__main__":
#     # Load the dataset
#     dataset = TextDataset('data.txt', 'data.csv')

#     # Define model parameters
#     vocab_size = 100  # Set according to your vocab size
#     d_model = 512
#     nhead = 8
#     num_encoder_layers = 6
#     num_decoder_layers = 6

#     # Create model instance
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

#     # Train the model
#     train(model, dataset, epochs=10, learning_rate=0.001)

#     # Generate text
#     output = generate_text(model, "Hello", max_length=20)
#     print("Generated Text:", output)
























































































































































# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim

# # Custom Dataset Class
# class TextDataset(Dataset):
#     def __init__(self, txt_file, csv_file):
#         self.txt_data = open(txt_file, 'r').readlines()
#         self.csv_data = pd.read_csv(csv_file)

#     def __len__(self):
#         return len(self.txt_data)

#     def __getitem__(self, idx):
#         text = self.txt_data[idx].strip()
#         additional_info = self.csv_data.iloc[idx]['text']
#         return text, additional_info

# # Transformer Model Class
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, src, tgt):
#         src = self.embedding(src)
#         tgt = self.embedding(tgt)
#         output = self.transformer(src, tgt)
#         return self.fc_out(output)

# # Training Function
# def train(model, dataset, epochs, learning_rate):
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             src, tgt = batch
#             src, tgt = src.to(device), tgt.to(device)

#             optimizer.zero_grad()
#             output = model(src, tgt)
#             loss = loss_fn(output.view(-1, output.size(-1)), tgt.view(-1))
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# # Text Generation Function
# def generate_text(model, input_text, max_length=50):
#     model.eval()
#     generated = input_text
#     input_tensor = torch.tensor([input_text]).unsqueeze(1).to(device)  # Convert input to tensor

#     for _ in range(max_length):
#         output = model(input_tensor, input_tensor)
#         next_word = output.argmax(-1)[-1].item()  # Get the most probable next word
#         generated += ' ' + str(next_word)  # Append the predicted word
#         input_tensor = torch.cat((input_tensor, torch.tensor([[next_word]]).to(device)), dim=0)  # Update input

#     return generated

# # Main Execution
# if __name__ == "__main__":
#     # Load the dataset
#     dataset = TextDataset('data.txt', 'data.csv')

#     # Define model parameters
#     vocab_size = 100  # Set according to your vocab size
#     d_model = 512
#     nhead = 8
#     num_encoder_layers = 6
#     num_decoder_layers = 6

#     # Create model instance
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

#     # Train the model
#     train(model, dataset, epochs=10, learning_rate=0.001)

#     # Generate text
#     output = generate_text(model, "Hello", max_length=20)
#     print("Generated Text:", output)






















































































































# import gradio as gr
# from huggingface_hub import InferenceClient

# client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")


# def respond(
#     message,
#     history: list[tuple[str, str]],
#     system_message,
#     max_tokens,
#     temperature,
#     top_p,
# ):
#     messages = [{"role": "system", "content": system_message}]

#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})

#     messages.append({"role": "user", "content": message})

#     response = ""

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content
        
#         if token is None:
#             token = ""
            
#         response += token
        
#         yield response

# demo = gr.ChatInterface(
#     respond,
#     additional_inputs=[
#         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#         gr.Slider(minimum=1, maximum=2048, value=2048, step=1, label="Max new tokens"),
#         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(
#             minimum=0.1,
#             maximum=1.0,
#             value=0.95,
#             step=0.05,
#             label="Top-p (nucleus sampling)",
#         ),
#     ],
# )


# if __name__ == "__main__":
#     demo.launch()































# import time
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# def fndriver():
#     options = webdriver.ChromeOptions()  
#     options.add_argument("--incognito")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_experimental_option("excludeSwitches", ["enable-automation"])
#     options.add_experimental_option('useAutomationExtension', False)
#     options.add_argument("--start-maximized")
    
#     driver = webdriver.Chrome(options=options)
#     driver.set_page_load_timeout(600) 

#     return driver

# # Create the driver instance
# driver = fndriver()

# try:
#     # Navigate to the website
#     driver.get('https://huggingface.co/spaces/AdamyaG/Open_gpt4O_fast')

#     # Wait for the button to be clickable and click it
#     # button = WebDriverWait(driver, 20).until(
#     #     EC.element_to_be_clickable((By.XPATH, '//*[@id="component-7"]//button'))  # Replace with the button's text or another identifier
#     # )
#     # button.click()

#     # Wait for the input field to be visible
#     input_field = WebDriverWait(driver, 20).until(
#         EC.visibility_of_element_located((By.XPATH, '//*[@data-testid="textbox"]//textarea'))  # Replace with the textarea's actual placeholder or another identifier
#     )
#     input_field.send_keys('Sample Data')  # Replace with the data you want to input
#     time.sleep(158888888888)
#     # # input_field.send_keys(Keys.RETURN)  # Submit if necessary
#     # # Wait for the result element to be visible
#     # result_element = WebDriverWait(driver, 20).until(
#     #     EC.visibility_of_element_located((By.XPATH, '//div[@class="result-class"]'))  # Replace with the actual class or identifier for the result element
#     # )
#     # print(result_element.text)

# except Exception as e:
#     print(f"An error occurred: {e}")

# finally:
#     # Close the WebDriver
#     driver.quit()
















































































# import json
# import os
# # import pandas as pd
# import json
# import time
# # import pymysql
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import Select
# import selenium.webdriver.support.ui as UI
# from selenium.common.exceptions import NoSuchElementException
# from datetime import datetime
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.action_chains import ActionChains

# def fndriver():
        
#         options = webdriver.ChromeOptions()  
#         options.add_argument("--incognito")
#         options.add_argument("--disable-blink-features=AutomationControlled")
#         options.add_experimental_option("excludeSwitches", ["enable-automation"])
#         options.add_experimental_option('useAutomationExtension', False)
#         options.add_argument("--start-maximized")
#         driver = webdriver.Chrome(options=options)
#         driver.set_page_load_timeout(600) 

#         return driver

# driver =  fndriver()
# driver.get("www.google.com")
































































# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.keys import Keys
# import time
# # Set up the WebDriver (using Chrome in this case)
# driver = webdriver.Chrome()

# try:
#     # Navigate to the website
#     driver.get('https://huggingface.co/spaces/AdamyaG/Open_gpt4O_fast')

#     # Wait for the button to be clickable and click it
#     button = WebDriverWait(driver, 20).until(
#         EC.element_to_be_clickable((By.XPATH, '//*[@id="component-7"]//button'))  # Replace with the button's text or another identifier
#     )
#     button.click()

#     # Wait for the input field to be visible
#     input_field = WebDriverWait(driver, 20).until(
#         EC.visibility_of_element_located((By.XPATH, '//*[@data-testid="textbox"]//textarea'))  # Replace with the textarea's actual placeholder or another identifier
#     )
#     input_field.send_keys('Sample Data')  # Replace with the data you want to input
#     time.sleep(158888888888)
#     # input_field.send_keys(Keys.RETURN)  # Submit if necessary
#     # Wait for the result element to be visible
#     result_element = WebDriverWait(driver, 20).until(
#         EC.visibility_of_element_located((By.XPATH, '//div[@class="result-class"]'))  # Replace with the actual class or identifier for the result element
#     )
#     print(result_element.text)

# except Exception as e:
#     print(f"An error occurred: {e}")

# finally:
#     # Close the WebDriver
#     driver.quit()






























# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# import time

# # Set up the WebDriver (using Chrome in this case)
# driver = webdriver.Chrome()

# try:
#     # Navigate to the website
#     driver.get('https://huggingface.co/spaces/YoMioAI/GPT-SoVITS-3s-cloning-free-TTS')  # Replace with the target URL
#     time.sleep(15)
#     # Click on an element using full XPath
#     button = driver.find_element(By.XPATH, '/html/body/gradio-app/div/div/div[1]/div/div/div[5]/div[2]/div/div[2]/div[2]/div/div/button[12]')  # Replace with the actual full XPath
#     button.click()
#     time.sleep(20000)
#     # Input data into a text field using full XPath
#     input_field = driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div/main/article/div/div[1]/section[1]/div/div/div/div[6]/div/div/div/div[3]/textarea')  # Replace with the actual full XPath
#     input_field.send_keys('Sample Data')  # Replace with the data you want to input
#     input_field.send_keys(Keys.RETURN)  # Submit if necessary

#     # Wait for a while to let the page update
#     time.sleep(200)

#     # Grab data from an element using full XPath
#     result_element = driver.find_element(By.XPATH, '/html/body/div[3]/div[1]')  # Replace with the actual full XPath
#     print(result_element.text)

# finally:
#     # Close the WebDriver
#     driver.quit()






# import asyncio
# from pyppeteer import launch
# import time

# async def main():
#     # Launch the browser
#     browser = await launch()
#     page = await browser.newPage()
#     # Navigate to the website
#     await page.goto('https://chatgpt4o.one')  # Replace with the target URL
#     # Click a button using XPath (example)
#     await page.click('xpath=/html/body/div[4]/div/div/div/main/article/div/div[1]/section[1]/div/div/div/div[6]/div/div/div/div[1]/span[2]/svg')  # Adjust the XPath as needed
#     time.sleep(3)
#     # Type into an input field
#     await page.type('xpath=/html/body/div[4]/div/div/div/main/article/div/div[1]/section[1]/div/div/div/div[6]/div/div/div/div[3]/textarea', 'hillow create a python code example')  # Adjust the XPath as needed
#     time.sleep(3)
#     # Submit the form or click a button to submit
#     await page.click('xpath=/html/body/div[4]/div/div/div/main/article/div/div[1]/section[1]/div/div/div/div[6]/div/div/div/div[3]/div/span[5]/svg')  # Adjust as needed
#     time.sleep(3)
#     # Wait for a selector to appear (e.g., result message)
#     await page.waitForSelector('xpath=/html/body/div[4]/div/div/div/main/article/div/div[1]/section[1]/div/div/div/div[6]/div/div/div/div[2]/ul/li[2]')  # Adjust as needed
#     time.sleep(3)
#     # Grab text from the result element
#     result_text = await page.evaluate('() => document.querySelector("xpath=/html/body/div[4]/div/div/div/main/article/div/div[1]/section[1]/div/div/div/div[6]/div/div/div/div[2]/ul/li[2]").innerText')
#     print(result_text)

#     # Close the browser
#     await browser.close()

# # Run the async function
# asyncio.get_event_loop().run_until_complete(main())
