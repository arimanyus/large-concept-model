import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional

class LCM(nn.Module):
    def __init__(self, 
                 encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
                 hidden_size: int = 768,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # Initialize sentence encoder
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        
        # Concept transformer layers
        self.concept_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Projection layers
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Geometric regularization
        self.lambda_reg = 0.1
        
    def _get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input_ids."""
        return (input_ids != self.tokenizer.pad_token_id).float()
    
    def _pool_embeddings(self, 
                        embeddings: torch.Tensor, 
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings to sentence embeddings using attention mask."""
        # Sum embeddings
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        summed = torch.sum(masked_embeddings, dim=1)
        
        # Get counts for averaging
        counts = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
        
        # Average
        pooled = summed / counts
        return pooled
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for training [batch_size, seq_len]
            
        Returns:
            outputs: Model outputs
            loss: Optional loss if labels are provided
        """
        # Get attention mask if not provided
        if attention_mask is None:
            attention_mask = self._get_attention_mask(input_ids)
            
        # Get embeddings from encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool to sentence embeddings
        sentence_embeddings = self._pool_embeddings(
            encoder_outputs.last_hidden_state,
            attention_mask
        )
        
        # Project to concept space
        concepts = self.input_projection(sentence_embeddings)
        
        # Apply concept transformer
        transformed = self.concept_transformer(
            concepts,
            src_key_padding_mask=None  # No padding at concept level
        )
        
        # Project back to token space
        outputs = self.output_projection(transformed)
        outputs = self.norm(outputs)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Get target embeddings
            with torch.no_grad():
                target_outputs = self.encoder(
                    input_ids=labels,
                    attention_mask=self._get_attention_mask(labels),
                    return_dict=True
                )
                target_embeddings = self._pool_embeddings(
                    target_outputs.last_hidden_state,
                    self._get_attention_mask(labels)
                )
            
            # MSE loss
            loss = nn.functional.mse_loss(outputs, target_embeddings)
            
            # Add geometric regularization
            reg_loss = self.lambda_reg * torch.mean(
                torch.norm(outputs - sentence_embeddings, dim=-1)
            )
            loss = loss + reg_loss
            
        return (outputs, loss) if loss is not None else (outputs,)
    
    def encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode sentences to concept embeddings."""
        self.eval()
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                all_embeddings.append(outputs[0])
                
        return torch.cat(all_embeddings, dim=0)
    
    def generate(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_length: int = 128,
                num_beams: int = 4,
                **kwargs) -> torch.Tensor:
        """Generate text from concept embeddings."""
        # Get concepts
        concepts = self(input_ids, attention_mask)[0]
        
        # Project back to token space
        token_logits = self.output_projection(concepts)
        
        # Use beam search for generation
        outputs = []
        batch_size = token_logits.size(0)
        
        for i in range(batch_size):
            # Initialize beams with start token
            beams = [([], 0.0)]  # (tokens, score)
            
            for _ in range(max_length):
                candidates = []
                
                for beam_tokens, beam_score in beams:
                    if len(beam_tokens) > 0 and beam_tokens[-1] == self.tokenizer.eos_token_id:
                        candidates.append((beam_tokens, beam_score))
                        continue
                        
                    # Get next token probabilities
                    logits = token_logits[i]  # Use concept embeddings
                    next_token_logits = torch.matmul(
                        logits,
                        self.encoder.embeddings.word_embeddings.weight.t()
                    )
                    
                    # Get top k tokens
                    values, indices = next_token_logits.topk(num_beams)
                    
                    for value, token in zip(values, indices):
                        new_tokens = beam_tokens + [token.item()]
                        new_score = beam_score + value.item()
                        candidates.append((new_tokens, new_score))
                
                # Select top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:num_beams]
                
                # Check if all beams ended
                if all(beam[0][-1] == self.tokenizer.eos_token_id for beam in beams):
                    break
            
            # Add best beam to outputs
            outputs.append(torch.tensor(beams[0][0]))
        
        return torch.stack(outputs) 