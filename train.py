import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm
import numpy as np
from model import LCM

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in train_bar:
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs[0], labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            wandb.log({
                'train_loss': loss.item(),
                'epoch': epoch
            })
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                loss = criterion(outputs[0], labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({
            'val_loss': avg_val_loss,
            'epoch': epoch
        })
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'final_model.pt')
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')

def main():
    # Initialize wandb
    wandb.init(project="lcm-practical", name="training-run")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Initialize model and tokenizer
    model = LCM()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    
    # Prepare data loaders
    def collate_fn(batch):
        # Tokenize and prepare batch
        inputs = tokenizer([item['article'] for item in batch], 
                         padding=True, 
                         truncation=True, 
                         return_tensors="pt")
        labels = tokenizer([item['highlights'] for item in batch],
                         padding=True,
                         truncation=True,
                         return_tensors="pt")['input_ids']
        inputs['labels'] = labels
        return inputs
    
    train_loader = DataLoader(dataset['train'], 
                            batch_size=8,
                            shuffle=True,
                            collate_fn=collate_fn)
    
    val_loader = DataLoader(dataset['validation'],
                          batch_size=8,
                          shuffle=False,
                          collate_fn=collate_fn)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train
    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, criterion)
    
    wandb.finish()

if __name__ == "__main__":
    main() 