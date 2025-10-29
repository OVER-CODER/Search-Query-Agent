#!/usr/bin/env python3
"""
Makefile functionality in Python for running tasks.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path


class TaskRunner:
    """Task runner for the Search Query Agent."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        
    def run_command(self, command: str, cwd: str = None, check: bool = True):
        """Run a shell command."""
        print(f"Running: {command}")
        if cwd:
            print(f"Working directory: {cwd}")
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or self.project_root,
            capture_output=False,
            check=check
        )
        
        return result.returncode == 0
    
    def install_dependencies(self):
        """Install Python dependencies."""
        print("=== Installing Dependencies ===")
        
        # Install main dependencies
        if not self.run_command("pip install -r requirements.txt"):
            print("Failed to install requirements. Trying with --user flag...")
            self.run_command("pip install --user -r requirements.txt")
        
        # Install Playwright browsers
        print("Installing Playwright browsers...")
        self.run_command("python -m playwright install", check=False)
        
        print("✓ Dependencies installed")
    
    def setup_environment(self):
        """Set up the development environment."""
        print("=== Setting Up Environment ===")
        
        # Create necessary directories
        dirs = [
            "training_data",
            "checkpoints", 
            "ollama_model",
            "logs",
            "browser_history"
        ]
        
        for dir_name in dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"Created directory: {dir_name}")
        
        # Copy environment file
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if env_example.exists() and not env_file.exists():
            import shutil
            shutil.copy2(env_example, env_file)
            print("Created .env file from .env.example")
        
        print("✓ Environment setup complete")
    
    def preview_history(self):
        """Run browser history preview."""
        print("=== Previewing Browser History ===")
        
        if not self.run_command("python scripts/preview_history.py"):
            print("Failed to preview browser history")
            return False
        
        return True
    
    def prepare_dataset(self):
        """Prepare training dataset."""
        print("=== Preparing Training Dataset ===")
        
        if not self.run_command("python scripts/prepare_dataset.py"):
            print("Failed to prepare dataset")
            return False
        
        print("✓ Dataset prepared")
        return True
    
    def train_model(self, steps: int = 10):
        """Train the QLoRA model."""
        print(f"=== Training Model ({steps} steps) ===")
        
        command = f"""python training/train_qlora.py \
            --output_dir ./checkpoints \
            --max_steps {steps} \
            --per_device_train_batch_size 1 \
            --learning_rate 2e-4 \
            --save_steps {steps//2} \
            --logging_steps 1"""
        
        if not self.run_command(command):
            print("Training failed. This might be expected without GPU or proper model access.")
            return False
        
        print("✓ Model training complete")
        return True
    
    def convert_model(self):
        """Convert model for Ollama."""
        print("=== Converting Model for Ollama ===")
        
        if not self.run_command("python training/convert_to_ollama.py --checkpoint-dir ./checkpoints"):
            print("Model conversion failed")
            return False
        
        print("✓ Model converted")
        return True
    
    def start_ollama(self):
        """Start Ollama server."""
        print("=== Starting Ollama Server ===")
        
        # Check if Ollama is installed
        if not self.run_command("ollama --version", check=False):
            print("Ollama not found. Please install Ollama first:")
            print("Visit: https://ollama.ai/download")
            return False
        
        # Start Ollama service (background)
        print("Starting Ollama service...")
        self.run_command("ollama serve &", check=False)
        
        # Give it time to start
        import time
        time.sleep(2)
        
        # Try to pull a model
        print("Attempting to pull llama3.1:8b model...")
        if self.run_command("ollama pull llama3.1:8b", check=False):
            print("✓ Model pulled successfully")
        else:
            print("Model pull failed. You may need to pull it manually:")
            print("ollama pull llama3.1:8b")
        
        return True
    
    def run_tests(self):
        """Run the test suite."""
        print("=== Running Tests ===")
        
        # Run tests without pytest if not available
        test_files = [
            "tests/test_dataset.py",
            "tests/test_training.py", 
            "tests/test_scraper.py",
            "tests/test_integration.py"
        ]
        
        passed = 0
        failed = 0
        
        for test_file in test_files:
            test_path = self.project_root / test_file
            if test_path.exists():
                print(f"\nRunning {test_file}...")
                if self.run_command(f"python {test_file}", check=False):
                    print(f"✓ {test_file} passed")
                    passed += 1
                else:
                    print(f"✗ {test_file} failed")
                    failed += 1
            else:
                print(f"⚠ {test_file} not found")
        
        print(f"\nTest Results: {passed} passed, {failed} failed")
        return failed == 0
    
    def start_server(self, port: int = 8000):
        """Start the API server."""
        print(f"=== Starting API Server on port {port} ===")
        
        os.environ["API_PORT"] = str(port)
        
        if not self.run_command(f"python app/main.py"):
            print("Failed to start server")
            return False
        
        return True
    
    def clean(self):
        """Clean up generated files."""
        print("=== Cleaning Up ===")
        
        import shutil
        
        cleanup_dirs = [
            "training_data",
            "checkpoints",
            "ollama_model", 
            "__pycache__",
            ".pytest_cache",
            "logs"
        ]
        
        for dir_name in cleanup_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"Removed: {dir_name}")
        
        # Remove .pyc files
        for pyc_file in self.project_root.rglob("*.pyc"):
            pyc_file.unlink()
        
        print("✓ Cleanup complete")
    
    def full_setup(self):
        """Run complete setup pipeline."""
        print("=== Full Setup Pipeline ===")
        
        steps = [
            ("Install dependencies", self.install_dependencies),
            ("Setup environment", self.setup_environment),
            ("Preview history", self.preview_history),
            ("Prepare dataset", self.prepare_dataset),
            ("Train model (micro)", lambda: self.train_model(5)),
            ("Convert model", self.convert_model),
            ("Run tests", self.run_tests),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*50}")
            print(f"Step: {step_name}")
            print('='*50)
            
            try:
                if not step_func():
                    print(f"⚠ Step '{step_name}' failed but continuing...")
            except Exception as e:
                print(f"⚠ Step '{step_name}' error: {e}")
        
        print("\n" + "="*50)
        print("Setup pipeline complete!")
        print("="*50)
        print("\nNext steps:")
        print("1. Start Ollama: make start-ollama")
        print("2. Start API server: make start-server")
        print("3. Test the API: curl http://localhost:8000/health")


def main():
    """Main CLI entry point."""
    runner = TaskRunner()
    
    if len(sys.argv) < 2:
        print("Usage: python make.py <command>")
        print("\nAvailable commands:")
        print("  install        - Install dependencies")
        print("  setup          - Setup environment")
        print("  preview        - Preview browser history")
        print("  dataset        - Prepare training dataset")
        print("  train          - Train model")
        print("  convert        - Convert model for Ollama")
        print("  start-ollama   - Start Ollama server")
        print("  test           - Run tests")
        print("  start-server   - Start API server")
        print("  clean          - Clean up files")
        print("  full-setup     - Run complete setup")
        return
    
    command = sys.argv[1]
    
    commands = {
        "install": runner.install_dependencies,
        "setup": runner.setup_environment,
        "preview": runner.preview_history,
        "dataset": runner.prepare_dataset,
        "train": runner.train_model,
        "convert": runner.convert_model,
        "start-ollama": runner.start_ollama,
        "test": runner.run_tests,
        "start-server": runner.start_server,
        "clean": runner.clean,
        "full-setup": runner.full_setup,
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()