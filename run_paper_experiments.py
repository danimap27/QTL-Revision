#!/usr/bin/env python3
"""
Script to run systematic experiments for the Quantum Transfer Learning paper.
Runs all models with 4 datasets and 4 pretrained architectures for 10 epochs.
"""

import os
import sys
import csv
import time
import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import uuid

# Imports from existing training functions
sys.path.append('.')
from train_cc import train_classical
from train_cq_qiskit import train_quantum_hybrid_qiskit
from train_cq_qiskit_noisy import train_quantum_hybrid_qiskit_noisy
from train_cq_pennylane import train_quantum_hybrid_pennylane
from train_cq_pennylane_noisy import train_quantum_hybrid_pennylane_noisy


# Experimental configurations for the paper
PAPER_DATASETS = ["hymenoptera", "brain_tumor", "cats_dogs", "solar_dust"]
PAPER_MODELS = ["resnet18", "resnet34", "vgg19", "mobilenetv2"] 
PAPER_EPOCHS = 10
PAPER_BATCH_SIZE = 16
PAPER_LEARNING_RATE = 0.001
PAPER_N_QUBITS = 4
PAPER_QUANTUM_DEPTH = 3

@dataclass
class ExperimentResult:
    """Result of an individual experiment."""
    model_type: str
    backbone: str
    dataset: str
    test_accuracy: float
    train_time_s: float
    test_time_s: float
    epochs: int
    run_id: str
    error_msg: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error_msg is None


class PaperExperimentsRunner:
    """Experiment runner for the paper."""
    
    def __init__(self, output_dir: str = "paper_results"):
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, "paper_experiments_results.csv")
        self.log_file = os.path.join(output_dir, "paper_experiments_log.txt")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize output files
        self._init_results_csv()
        self._init_log()
        
    def _init_results_csv(self):
        """Initialize the results CSV file."""
        fieldnames = [
            'Timestamp', 'Model_Type', 'Backbone', 'Dataset', 
            'Test_Accuracy', 'Train_Time_s', 'Test_Time_s', 'Epochs',
            'Batch_Size', 'Learning_Rate', 'N_Qubits', 'Quantum_Depth',
            'Run_ID', 'Status', 'Error_Message'
        ]
        
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
    def _init_log(self):
        """Initialize the log file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== QUANTUM TRANSFER LEARNING PAPER EXPERIMENTS ===\n")
            f.write(f"Start: {datetime.datetime.now()}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Datasets: {PAPER_DATASETS}\n")
            f.write(f"  - Models: {PAPER_MODELS}\n")
            f.write(f"  - Epochs: {PAPER_EPOCHS}\n")
            f.write(f"  - Batch Size: {PAPER_BATCH_SIZE}\n")
            f.write(f"  - Learning Rate: {PAPER_LEARNING_RATE}\n")
            f.write(f"  - Qubits: {PAPER_N_QUBITS}\n")
            f.write(f"  - Quantum Depth: {PAPER_QUANTUM_DEPTH}\n")
            f.write("="*60 + "\n\n")

    def log(self, message: str):
        """Write a message to log and console."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        print(full_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(full_msg + "\n")
            
    def save_result(self, result: ExperimentResult):
        """Save an individual result to CSV."""
        row = {
            'Timestamp': datetime.datetime.now().isoformat(),
            'Model_Type': result.model_type,
            'Backbone': result.backbone,
            'Dataset': result.dataset,
            'Test_Accuracy': f"{result.test_accuracy:.4f}" if result.success else "ERROR",
            'Train_Time_s': f"{result.train_time_s:.2f}" if result.success else "ERROR",
            'Test_Time_s': f"{result.test_time_s:.2f}" if result.success else "ERROR", 
            'Epochs': result.epochs,
            'Batch_Size': PAPER_BATCH_SIZE,
            'Learning_Rate': PAPER_LEARNING_RATE,
            'N_Qubits': PAPER_N_QUBITS if 'quantum' in result.model_type.lower() else "N/A",
            'Quantum_Depth': PAPER_QUANTUM_DEPTH if 'quantum' in result.model_type.lower() else "N/A",
            'Run_ID': result.run_id,
            'Status': "SUCCESS" if result.success else "ERROR",
            'Error_Message': result.error_msg or ""
        }
        
        with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def run_single_experiment(self, model_type: str, backbone: str, dataset: str) -> ExperimentResult:
        """Run an individual experiment."""
        run_id = f"{model_type}_{backbone}_{dataset}_{uuid.uuid4().hex[:8]}"
        
        self.log(f"🚀 Starting: {model_type} | {backbone} | {dataset} (ID: {run_id})")
        
        try:
            if model_type == "Classical":
                acc, train_time, test_time = train_classical(
                    dataset_file=dataset,
                    classical_model=backbone,
                    epochs=PAPER_EPOCHS,
                    id=run_id,
                    batch_size=PAPER_BATCH_SIZE,
                    learning_rate=PAPER_LEARNING_RATE
                )
                
            elif model_type == "Qiskit_Standard":
                acc, train_time, test_time = train_quantum_hybrid_qiskit(
                    dataset_file=dataset,
                    classical_model=backbone,
                    n_qubits=PAPER_N_QUBITS,
                    quantum_depth=PAPER_QUANTUM_DEPTH,
                    epochs=PAPER_EPOCHS,
                    id=run_id,
                    batch_size=PAPER_BATCH_SIZE,
                    learning_rate=PAPER_LEARNING_RATE,
                    gamma=0.9
                )
                
            elif model_type == "Qiskit_Noisy":
                acc, train_time, test_time = train_quantum_hybrid_qiskit_noisy(
                    dataset_file=dataset,
                    classical_model=backbone,
                    n_qubits=PAPER_N_QUBITS,
                    quantum_depth=PAPER_QUANTUM_DEPTH,
                    epochs=PAPER_EPOCHS,
                    id=run_id,
                    batch_size=PAPER_BATCH_SIZE,
                    learning_rate=PAPER_LEARNING_RATE,
                    backend_name="ibm_nairobi"  # Default realistic IBM backend
                )
                
            elif model_type == "PennyLane_Standard":
                acc, train_time, test_time = train_quantum_hybrid_pennylane(
                    dataset_file=dataset,
                    classical_model=backbone,
                    n_qubits=PAPER_N_QUBITS,
                    epochs=PAPER_EPOCHS,
                    id=run_id,
                    batch_size=PAPER_BATCH_SIZE,
                    learning_rate=PAPER_LEARNING_RATE,
                    early_stop_patience=15
                )
                
            elif model_type == "PennyLane_Noisy":
                acc, train_time, test_time = train_quantum_hybrid_pennylane_noisy(
                    dataset_file=dataset,
                    classical_model=backbone,
                    n_qubits=PAPER_N_QUBITS,
                    epochs=PAPER_EPOCHS,
                    id=run_id,
                    batch_size=PAPER_BATCH_SIZE,
                    learning_rate=PAPER_LEARNING_RATE,
                    backend="ibm_nairobi"  # Default realistic IBM backend
                )
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            result = ExperimentResult(
                model_type=model_type,
                backbone=backbone,
                dataset=dataset,
                test_accuracy=acc,
                train_time_s=train_time,
                test_time_s=test_time,
                epochs=PAPER_EPOCHS,
                run_id=run_id
            )
            
            self.log(f"✅ SUCCESS: {model_type} | {backbone} | {dataset} -> Acc: {acc:.2%}, Train: {train_time:.1f}s")
            
        except Exception as e:
            error_msg = str(e)
            result = ExperimentResult(
                model_type=model_type,
                backbone=backbone,
                dataset=dataset,
                test_accuracy=0.0,
                train_time_s=0.0,
                test_time_s=0.0,
                epochs=PAPER_EPOCHS,
                run_id=run_id,
                error_msg=error_msg
            )
            
            self.log(f"❌ ERROR: {model_type} | {backbone} | {dataset} -> {error_msg}")
            
        return result

    def run_all_experiments(self, approaches: List[str] = None):
        """Run all experiments for the paper."""
        
        # Define all available approaches
        all_approaches = {
            "classical": "Classical",
            "qiskit": "Qiskit_Standard", 
            "qiskit_noisy": "Qiskit_Noisy",
            "pennylane": "PennyLane_Standard",
            "pennylane_noisy": "PennyLane_Noisy"
        }
        
        # Use all approaches if none are specified
        if approaches is None:
            approaches = list(all_approaches.keys())
            
        # Filter selected approaches
        selected_approaches = {k: v for k, v in all_approaches.items() if k in approaches}
        
        total_experiments = len(selected_approaches) * len(PAPER_MODELS) * len(PAPER_DATASETS)
        current_exp = 0
        
        self.log(f"🔬 Starting {total_experiments} experiments for the paper")
        self.log(f"📋 Selected approaches: {list(selected_approaches.keys())}")
        
        start_time = time.time()
        results = []
        
        # Run experiments by approach -> model -> dataset
        for approach_key, model_type in selected_approaches.items():
            self.log(f"\n🔬 === APPROACH: {model_type} ===")
            
            for backbone in PAPER_MODELS:
                self.log(f"\n🏗️  Backbone: {backbone}")
                
                for dataset in PAPER_DATASETS:
                    current_exp += 1
                    progress = f"[{current_exp}/{total_experiments}]"
                    
                    self.log(f"{progress} Running {model_type} + {backbone} + {dataset}")
                    
                    result = self.run_single_experiment(model_type, backbone, dataset)
                    results.append(result)
                    self.save_result(result)
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / current_exp if current_exp > 0 else 0
                    eta = avg_time * (total_experiments - current_exp)
                    
                    self.log(f"{progress} Completed in {elapsed:.1f}s, ETA: {eta:.1f}s")
                    
        # Final summary
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        self.log(f"\n🎯 === FINAL SUMMARY ===")
        self.log(f"✅ Successful experiments: {successful}/{total_experiments}")
        self.log(f"❌ Failed experiments: {failed}/{total_experiments}")
        self.log(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        self.log(f"📊 Results saved in: {self.results_file}")
        
        # Show best results by approach
        self._show_best_results(results)
        
        return results
    
    def _show_best_results(self, results: List[ExperimentResult]):
        """Show the best results by approach."""
        self.log(f"\n🏆 === BEST RESULTS BY APPROACH ===")
        
        # Group by model type
        by_model_type = {}
        for result in results:
            if result.success:
                if result.model_type not in by_model_type:
                    by_model_type[result.model_type] = []
                by_model_type[result.model_type].append(result)
        
        # Show the best result of each approach
        for model_type, model_results in by_model_type.items():
            best = max(model_results, key=lambda r: r.test_accuracy)
            self.log(f"🥇 {model_type}: {best.test_accuracy:.2%} ({best.backbone} + {best.dataset})")
            
        # Show the overall best result
        if results:
            successful_results = [r for r in results if r.success]
            if successful_results:
                overall_best = max(successful_results, key=lambda r: r.test_accuracy)
                self.log(f"\n🌟 OVERALL BEST: {overall_best.test_accuracy:.2%}")
                self.log(f"   Model: {overall_best.model_type}")
                self.log(f"   Backbone: {overall_best.backbone}")
                self.log(f"   Dataset: {overall_best.dataset}")
                self.log(f"   Time: {overall_best.train_time_s:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Experiment runner for the paper")
    parser.add_argument("--approaches", nargs="+", 
                        choices=["classical", "qiskit", "qiskit_noisy", "pennylane", "pennylane_noisy"],
                        help="Approaches to run (default: all)")
    parser.add_argument("--output-dir", default="paper_results",
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would run, without executing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY-RUN MODE: Showing experimental configuration")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🔬 Approaches: {args.approaches or 'ALL'}")
        print(f"📊 Datasets: {PAPER_DATASETS}")
        print(f"🏗️  Models: {PAPER_MODELS}")
        print(f"⚙️  Epochs: {PAPER_EPOCHS}")
        print(f"📦 Batch Size: {PAPER_BATCH_SIZE}")
        print(f"📈 Learning Rate: {PAPER_LEARNING_RATE}")
        print(f"⚛️  Qubits: {PAPER_N_QUBITS}")
        print(f"🔄 Quantum Depth: {PAPER_QUANTUM_DEPTH}")
        
        approaches = args.approaches or ["classical", "qiskit", "qiskit_noisy", "pennylane", "pennylane_noisy"]
        total = len(approaches) * len(PAPER_MODELS) * len(PAPER_DATASETS)
        print(f"\n📊 Total experiments to run: {total}")
        print("✅ Configuration verified. Run without --dry-run to start.")
        return
    
    # Run full experiments
    runner = PaperExperimentsRunner(output_dir=args.output_dir)
    
    try:
        results = runner.run_all_experiments(approaches=args.approaches)
        
        print(f"\n🎉 Experiments completed successfully!")
        print(f"📊 Results available at: {runner.results_file}")
        print(f"📋 Full log available at: {runner.log_file}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiments interrupted by user")
        print(f"📊 Partial results at: {runner.results_file}")
        
    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")
        raise


if __name__ == "__main__":
    main()