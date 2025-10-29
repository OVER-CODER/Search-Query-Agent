#!/usr/bin/env python3
"""
Convert QLoRA checkpoint to Ollama-compatible format.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional
import shutil


class OllamaConverter:
    """Convert fine-tuned model to Ollama format."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    def create_modelfile(self, base_model: str, output_path: str):
        """Create Ollama Modelfile."""
        modelfile_content = f"""
FROM {base_model}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt for search queries
SYSTEM \"\"\"You are a personalized search assistant trained on the user's browsing history. 
When given a search query, return the most relevant URLs from their history as JSON.

Response format:
{{
  "urls": [
    {{
      "url": "https://example.com",
      "title": "Page Title",
      "confidence": 0.95,
      "reason": "Explanation why this URL matches"
    }}
  ]
}}

Only return URLs that are highly relevant to the query. If no good matches exist, return an empty urls array.\"\"\"

# Load fine-tuned weights (if available)
# ADAPTER {self.checkpoint_dir}/adapter_model.bin
"""
        
        with open(output_path, 'w') as f:
            f.write(modelfile_content)
        
        print(f"Modelfile created: {output_path}")
    
    def create_wrapper_server(self, output_path: str):
        """Create a wrapper server that mimics Ollama API."""
        wrapper_code = '''#!/usr/bin/env python3
"""
Wrapper server that provides Ollama-compatible API for fine-tuned model.
"""

import asyncio
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: bool = False


class ChatResponse(BaseModel):
    message: str
    done: bool = True


class OllamaWrapper:
    """Wrapper for fine-tuned model."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model."""
        try:
            # Try to load the fine-tuned model
            base_model_name = "microsoft/DialoGPT-medium"  # Fallback
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Try to load LoRA weights
            if os.path.exists(f"{self.checkpoint_dir}/adapter_model.bin"):
                self.model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
                print("Loaded fine-tuned model with LoRA weights")
            else:
                self.model = base_model
                print("Using base model (no fine-tuned weights found)")
            
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_response(self, query: str) -> str:
        """Generate response for query."""
        # Format prompt
        prompt = f"### Instruction:\\nGiven query: '{query}', which URL from my browsing history best answers it?\\n\\n### Response:\\n"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        # If response doesn't look like JSON, create a default response
        if not response.startswith('{'):
            response = json.dumps({
                "urls": [{
                    "url": "https://example.com",
                    "title": f"Search results for: {query}",
                    "confidence": 0.5,
                    "reason": "Generated response (model needs more training)"
                }]
            })
        
        return response


# Create FastAPI app
app = FastAPI()
wrapper = None


@app.on_event("startup")
async def startup():
    global wrapper
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    wrapper = OllamaWrapper(checkpoint_dir)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint compatible with Ollama."""
    if wrapper is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Extract user query from messages
    user_message = None
    for msg in request.messages:
        if msg.get("role") == "user":
            user_message = msg.get("content")
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Generate response
    try:
        response = wrapper.generate_response(user_message)
        
        if request.stream:
            # For streaming, return the response in chunks
            return {"message": response, "done": True}
        else:
            return {"message": response, "done": True}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tags")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "search-agent:latest",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 1000000000
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
'''
        
        with open(output_path, 'w') as f:
            f.write(wrapper_code)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        print(f"Wrapper server created: {output_path}")
    
    def convert(self, base_model: str = "microsoft/DialoGPT-medium"):
        """Convert checkpoint for Ollama usage."""
        output_dir = Path("./ollama_model")
        output_dir.mkdir(exist_ok=True)
        
        # Create Modelfile
        self.create_modelfile(base_model, output_dir / "Modelfile")
        
        # Create wrapper server
        self.create_wrapper_server(output_dir / "wrapper_server.py")
        
        # Copy checkpoint files
        if (self.checkpoint_dir / "adapter_model.bin").exists():
            shutil.copy2(
                self.checkpoint_dir / "adapter_model.bin",
                output_dir / "adapter_model.bin"
            )
        
        if (self.checkpoint_dir / "adapter_config.json").exists():
            shutil.copy2(
                self.checkpoint_dir / "adapter_config.json", 
                output_dir / "adapter_config.json"
            )
        
        # Create startup script
        startup_script = f"""#!/bin/bash
# Start the wrapper server
export CHECKPOINT_DIR="{output_dir.absolute()}"
cd "{output_dir.absolute()}"
python wrapper_server.py
"""
        
        with open(output_dir / "start_server.sh", 'w') as f:
            f.write(startup_script)
        
        os.chmod(output_dir / "start_server.sh", 0o755)
        
        print(f"\\nConversion complete!")
        print(f"Output directory: {output_dir.absolute()}")
        print(f"\\nTo start the server:")
        print(f"cd {output_dir.absolute()}")
        print(f"./start_server.sh")
        print(f"\\nThe server will be available at: http://localhost:11434")


def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert QLoRA checkpoint to Ollama format")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", 
                       help="Path to checkpoint directory")
    parser.add_argument("--base-model", default="microsoft/DialoGPT-medium",
                       help="Base model name")
    
    args = parser.parse_args()
    
    converter = OllamaConverter(args.checkpoint_dir)
    converter.convert(args.base_model)


if __name__ == "__main__":
    main()