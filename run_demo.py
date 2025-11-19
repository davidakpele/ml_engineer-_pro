#!/usr/bin/env python3
"""
Complete ML Project Demo Runner
"""
import sys
import os
import subprocess
import time

def run_command(command, description):
    """Run a shell command with description"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def main():
    """Run the complete ML project demo"""
    print("🤖 ML Engineer Portfolio - Complete Demo")
    print("This will run the entire ML pipeline from data generation to API deployment")
    
    # Step 1: Generate sample data
    if not run_command("python scripts/generate_sample_data.py", "Generating sample data"):
        return
    
    # Step 2: Train models
    if not run_command("python scripts/train_model.py", "Training ML models"):
        return
    
    # Step 3: Run drift monitoring
    if not run_command("python scripts/monitor_drift.py", "Running drift monitoring"):
        return
    
    # Step 4: Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        return
    
    # Step 5: Start API (in background)
    print(f"\n{'='*60}")
    print("🚀 Starting ML API Server")
    print(f"{'='*60}")
    print("The API will start on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start the API server
        subprocess.run(["python", "scripts/deploy_model.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo completed! API server stopped.")
    
    print("\n🎉 Demo completed successfully!")
    print("You can now:")
    print("  - Visit http://localhost:8000 for API docs")
    print("  - Visit http://localhost:8000/health for health check")
    print("  - Use /predict endpoint for predictions")
    print("  - Check logs/ directory for logs and monitoring reports")

if __name__ == "__main__":
    main()