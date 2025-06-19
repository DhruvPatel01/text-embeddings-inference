#!/usr/bin/env python3
"""
Test script to verify Qwen3 reranker/classification support works.
"""

import subprocess
import sys
import time
import requests
import json

def test_qwen3_reranker():
    """Test the Qwen3 reranker model."""
    
    # Start the server with Qwen3 reranker model
    model_id = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
    
    print(f"Testing Qwen3 reranker model: {model_id}")
    
    # Start the server
    cmd = [
        "text-embeddings-router",
        "--model-id", model_id,
        "--port", "8081",
        "--hostname", "0.0.0.0"
    ]
    
    print(f"Starting server with command: {' '.join(cmd)}")
    
    try:
        # Start the server process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(10)
        
        # Test reranking endpoint
        rerank_data = {
            "query": "What is deep learning?",
            "texts": [
                "Deep learning is a subset of machine learning that uses neural networks.",
                "Machine learning is a field of artificial intelligence.",
                "Python is a programming language.",
                "Neural networks are computational models inspired by biological neural networks."
            ],
            "raw_scores": False,
            "return_text": True
        }
        
        print("Testing reranking endpoint...")
        response = requests.post(
            "http://localhost:8081/rerank",
            json=rerank_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Reranking successful!")
            print(f"Number of results: {len(result)}")
            for i, rank in enumerate(result[:2]):  # Show top 2
                print(f"  {i+1}. Score: {rank['score']:.4f}, Text: {rank['text'][:50]}...")
            return True
        else:
            print(f"❌ Reranking failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing reranker: {e}")
        return False
    finally:
        # Clean up
        if 'process' in locals():
            process.terminate()
            process.wait(timeout=5)
            print("Server stopped.")

if __name__ == "__main__":
    success = test_qwen3_reranker()
    sys.exit(0 if success else 1)