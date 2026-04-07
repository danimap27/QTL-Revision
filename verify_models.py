#!/usr/bin/env python3
"""
Quick verification script to check that all models run correctly.
Runs 1 epoch on 1 dataset to validate the setup before full experiments.
"""

import sys
import time

# Import training functions
sys.path.append('.')
from train_cc import train_classical
from train_cq_qiskit import train_quantum_hybrid_qiskit
from train_cq_qiskit_noisy import train_quantum_hybrid_qiskit_noisy
from train_cq_pennylane import train_quantum_hybrid_pennylane
from train_cq_pennylane_noisy import train_quantum_hybrid_pennylane_noisy

def test_single_model(func, name, **kwargs):
    """Test an individual model with a minimal configuration."""
    print(f"🔍 Testing {name}... ", end="", flush=True)
    
    start_time = time.time()
    try:
        # Minimal test configuration
        acc, train_time, test_time = func(
            dataset_file="hymenoptera",  # Smallest dataset
            classical_model="resnet18",   # Fastest model
            epochs=1,                     # Single epoch
            id=f"test_{name}",
            batch_size=8,                 # Small batch for speed
            learning_rate=0.001,
            **kwargs
        )
        
        elapsed = time.time() - start_time
        print(f"✅ OK | Acc: {acc:.2%} | {elapsed:.1f}s")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR: {str(e)[:60]}... | {elapsed:.1f}s")
        return False

def verify_all_models():
    """Verify that all models run correctly."""
    print("🔧 QUICK MODEL VERIFICATION")
    print("="*60)
    print("Dataset: hymenoptera | Model: resnet18 | Epochs: 1")
    print("="*60)
    
    # Test configurations
    tests = [
        (train_classical, "Classical", {}),
        (train_quantum_hybrid_qiskit, "Qiskit_Perfect", {"n_qubits": 4, "quantum_depth": 3, "gamma": 0.9}),
        (train_quantum_hybrid_qiskit_noisy, "Qiskit_Noisy", {"n_qubits": 4, "quantum_depth": 3, "backend": "ibm_nairobi"}),
        (train_quantum_hybrid_pennylane, "PennyLane_Perfect", {"n_qubits": 4, "early_stop_patience": 5}),
        (train_quantum_hybrid_pennylane_noisy, "PennyLane_Noisy", {"n_qubits": 4, "backend": "ibm_nairobi"})
    ]
    
    results = []
    total_time = 0
    
    for func, name, params in tests:
        start = time.time()
        success = test_single_model(func, name, **params)
        elapsed = time.time() - start
        total_time += elapsed
        results.append((name, success))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ OK" if success else "❌ FAILED"
        print(f"  {status} {name}")
    
    print(f"\n🎯 Result: {successful}/{total} models running correctly")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if successful == total:
        print("\n🚀 All models verified. Ready for full experiments.")
        print("\nRun full experiments with:")
        print("  python run_paper_simple.py          # Simple version")  
        print("  python run_paper_experiments.py     # Full version with CSV")
    else:
        print(f"\n⚠️  {total-successful} models failed. Review the setup before proceeding.")
    
    return successful == total

if __name__ == "__main__":
    try:
        verify_all_models()
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during verification: {e}")
        raise