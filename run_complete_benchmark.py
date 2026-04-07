"""run_complete_benchmark.py - Run the full benchmark of 5 architectures × 4 datasets × 4 backbones.

This script automatically executes:
- 5 architectures: CC, PennyLane, PennyLane-Noisy, Qiskit, Qiskit-Noisy
- 4 datasets: hymenoptera, brain_tumor, cats_dogs, solar_dust
- 4 backbones: resnet18, mobilenetv2, efficientnet_b0, regnet_x_400mf

Total: 80 experiments (5 × 4 × 4)

Generates:
- Trained models (.pth)
- Metrics (accuracy, precision, recall, F1, AUC)
- Plots (loss, confusion matrix, ROC)
- Consolidated CSV with all results
- Comparative visualizations

Uso:
    python run_complete_benchmark.py --epochs 10 --batch-size 16
    python run_complete_benchmark.py --quick-test  # Only 1 epoch for testing
    python run_complete_benchmark.py --datasets hymenoptera solar_dust  # Only selected datasets
"""

import subprocess
import time
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Configuration
DATASETS = ['hymenoptera', 'brain_tumor', 'cats_dogs', 'solar_dust']
BACKBONES = ['resnet18', 'mobilenetv2', 'efficientnet_b0', 'regnet_x_400mf']
ARCHITECTURES = ['classical', 'pennylane', 'pennylane_noisy', 'qiskit', 'qiskit_noisy']

# Training scripts
TRAIN_SCRIPTS = {
    'classical': 'train_cc.py',
    'pennylane': 'train_cq_pennylane.py',
    'pennylane_noisy': 'train_cq_pennylane_noisy.py',
    'qiskit': 'train_cq_qiskit.py',
    'qiskit_noisy': 'train_cq_qiskit_noisy.py'
}


class BenchmarkRunner:
    """Run and manage the full benchmark."""
    
    def __init__(self, epochs=10, batch_size=16, n_qubits=4, depth=3, 
                 learning_rate=1e-3, gamma=0.9, shots=None, noise_1q=0.001, noise_2q=0.01):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_qubits = n_qubits
        self.depth = depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.shots = shots
        self.noise_1q = noise_1q
        self.noise_2q = noise_2q
        
        self.results = []
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.results_dir = Path("benchmark_results") / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
    
    def run_experiment(self, architecture, dataset, backbone):
        """Run an individual experiment."""
        exp_id = f"{architecture}_{backbone}_{dataset}_{self.run_id}"
        log_file = self.results_dir / "logs" / f"{exp_id}.log"
        
        print(f"\n{'='*80}")
        print(f"Experiment: {architecture} | {backbone} | {dataset}")
        print(f"{'='*80}")
        
        # Build command
        script = TRAIN_SCRIPTS[architecture]
        cmd = [
            'python', script,
            '--dataset', dataset,
            '--model', backbone,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--lr', str(self.learning_rate),
            '--id', exp_id
        ]
        
        # Gamma applies only to quantum models (not available in train_cc.py)
        if architecture in ['pennylane', 'pennylane_noisy', 'qiskit', 'qiskit_noisy']:
            cmd.extend(['--gamma', str(self.gamma)])
        
        # Quantum-specific parameters
        if architecture in ['pennylane', 'pennylane_noisy', 'qiskit', 'qiskit_noisy']:
            cmd.extend(['--n-qubits', str(self.n_qubits)])
            cmd.extend(['--depth', str(self.depth)])
        
        # Shots parameters for Qiskit (noise is already integrated in qiskit_noisy)
        if architecture in ['qiskit', 'qiskit_noisy'] and self.shots:
            cmd.extend(['--shots', str(self.shots)])
        
        # Execute
        t0 = time.time()
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=7200  # 2 hours maximum
                )
            
            elapsed = time.time() - t0
            success = (result.returncode == 0)
            
            if success:
                print(f"✅ Completed in {elapsed:.1f}s")
            else:
                print(f"❌ Error (code {result.returncode})")
            
            # Parse results from log
            metrics = self._parse_log(log_file)
            
            # Save result
            self.results.append({
                'architecture': architecture,
                'dataset': dataset,
                'backbone': backbone,
                'exp_id': exp_id,
                'success': success,
                'elapsed_time': elapsed,
                'epochs': self.epochs,
                **metrics
            })
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"⏱️ Timeout after 2 hours")
            elapsed = time.time() - t0
            self.results.append({
                'architecture': architecture,
                'dataset': dataset,
                'backbone': backbone,
                'exp_id': exp_id,
                'success': False,
                'elapsed_time': elapsed,
                'error': 'timeout'
            })
            return False
        
        except Exception as e:
            print(f"❌ Exception: {e}")
            self.results.append({
                'architecture': architecture,
                'dataset': dataset,
                'backbone': backbone,
                'exp_id': exp_id,
                'success': False,
                'error': str(e)
            })
            return False
    
    def _parse_log(self, log_file):
        """Extract metrics from a training log."""
        metrics = {}
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Find test accuracy
            if 'Test acc=' in log_content:
                for line in log_content.split('\n'):
                    if 'Test acc=' in line:
                        # Format: "Test acc=0.9487 test_time=2.44s train_time=30.90s"
                        parts = line.split()
                        for part in parts:
                            if part.startswith('acc='):
                                metrics['test_accuracy'] = float(part.split('=')[1])
                            elif part.startswith('test_time='):
                                metrics['test_time'] = float(part.split('=')[1].rstrip('s'))
                            elif part.startswith('train_time='):
                                metrics['train_time'] = float(part.split('=')[1].rstrip('s'))
            
            # Find additional metrics (Precision, Recall, F1)
            if 'Precision:' in log_content:
                for line in log_content.split('\n'):
                    if 'Precision:' in line:
                        # Format: "Precision: 0.9500 Recall: 0.9487 F1: 0.9489"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'Precision:' and i+1 < len(parts):
                                metrics['precision'] = float(parts[i+1])
                            elif part == 'Recall:' and i+1 < len(parts):
                                metrics['recall'] = float(parts[i+1])
                            elif part == 'F1:' and i+1 < len(parts):
                                metrics['f1'] = float(parts[i+1])
            
            # Find AUC
            if 'AUC:' in log_content:
                for line in log_content.split('\n'):
                    if 'AUC:' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'AUC:' and i+1 < len(parts):
                                metrics['auc'] = float(parts[i+1])
        
        except Exception as e:
            print(f"⚠️  Error parsing log: {e}")
        
        return metrics
    
    def run_all(self, datasets=None, architectures=None, backbones=None):
        """Run all experiments."""
        datasets = datasets or DATASETS
        architectures = architectures or ARCHITECTURES
        backbones = backbones or BACKBONES
        
        total = len(datasets) * len(architectures) * len(backbones)
        current = 0
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETO")
        print(f"{'='*80}")
        print(f"Datasets: {datasets}")
        print(f"Architectures: {architectures}")
        print(f"Backbones: {backbones}")
        print(f"Total experiments: {total}")
        print(f"Epochs per experiment: {self.epochs}")
        print(f"Estimated time: ~{total * 5} minutes (5 min/exp average)")
        print(f"{'='*80}\n")
        
        for dataset in datasets:
            for backbone in backbones:
                for architecture in architectures:
                    current += 1
                    print(f"\n[{current}/{total}] Progress: {current/total*100:.1f}%")
                    self.run_experiment(architecture, dataset, backbone)
                    
                    # Save partial results
                    self.save_results()
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETED")
        print(f"{'='*80}")
        print(f"Successful experiments: {sum(r['success'] for r in self.results)}/{total}")
        print(f"Total time: {(datetime.now() - self.start_time).total_seconds()/60:.1f} min")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*80}\n")
    
    def save_results(self):
        """Save results to CSV and JSON."""
        df = pd.DataFrame(self.results)
        
        # CSV
        csv_path = self.results_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        
        # JSON
        json_path = self.results_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📊 Results saved: {csv_path}")
    
    def generate_visualizations(self):
        """Generate comparative visualizations."""
        df = pd.DataFrame(self.results)
        
        if df.empty or 'test_accuracy' not in df.columns:
            print("⚠️  Not enough data available for visualizations")
            return
        
        # Filter successful experiments only
        df = df[df['success'] == True].copy()
        
        # 1. Accuracy heatmap by Dataset × Architecture
        plt.figure(figsize=(12, 8))
        pivot = df.pivot_table(
            values='test_accuracy',
            index='architecture',
            columns='dataset',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0)
        plt.title('Test Accuracy: Architecture × Dataset (promedio sobre backbones)')
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "heatmap_accuracy_dataset.png", dpi=150)
        plt.close()
        
        # 2. Accuracy heatmap by Backbone × Architecture
        plt.figure(figsize=(12, 8))
        pivot = df.pivot_table(
            values='test_accuracy',
            index='architecture',
            columns='backbone',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0)
        plt.title('Test Accuracy: Architecture × Backbone (promedio sobre datasets)')
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "heatmap_accuracy_backbone.png", dpi=150)
        plt.close()
        
        # 3. Training time bar plot by Architecture
        if 'train_time' in df.columns:
            plt.figure(figsize=(10, 6))
            train_time = df.groupby('architecture')['train_time'].mean().sort_values()
            train_time.plot(kind='barh', color='steelblue')
            plt.xlabel('Training Time (seconds)')
            plt.title('Average Training Time by Architecture')
            plt.tight_layout()
            plt.savefig(self.results_dir / "plots" / "training_time_comparison.png", dpi=150)
            plt.close()
        
        # 4. Scatter: Accuracy vs Training Time
        if 'train_time' in df.columns:
            plt.figure(figsize=(10, 6))
            for arch in df['architecture'].unique():
                data = df[df['architecture'] == arch]
                plt.scatter(
                    data['train_time'],
                    data['test_accuracy'],
                    label=arch,
                    alpha=0.6,
                    s=100
                )
            plt.xlabel('Training Time (s)')
            plt.ylabel('Test Accuracy')
            plt.title('Accuracy vs Training Time Trade-off')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / "plots" / "accuracy_vs_time.png", dpi=150)
            plt.close()
        
        # 5. Accuracy boxplot by Architecture
        plt.figure(figsize=(12, 6))
        df.boxplot(column='test_accuracy', by='architecture', figsize=(12, 6))
        plt.suptitle('')
        plt.title('Test Accuracy Distribution by Architecture')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "boxplot_accuracy.png", dpi=150)
        plt.close()
        
        print(f"📊 Visualizations generated in: {self.results_dir}/plots/")
    
    def generate_summary_report(self):
        """Generate a summary report in markdown."""
        df = pd.DataFrame(self.results)
        df_success = df[df['success'] == True].copy()
        
        report_path = self.results_dir / "SUMMARY.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Benchmark Summary Report\n\n")
            f.write(f"**Run ID:** {self.run_id}\n")
            f.write(f"**Date:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Runtime:** {(datetime.now() - self.start_time).total_seconds()/60:.1f} min\n\n")
            
            f.write(f"## Configuration\n\n")
            f.write(f"- Epochs: {self.epochs}\n")
            f.write(f"- Batch Size: {self.batch_size}\n")
            f.write(f"- Learning Rate: {self.learning_rate}\n")
            f.write(f"- Quantum Qubits: {self.n_qubits}\n")
            f.write(f"- Quantum Depth: {self.depth}\n\n")
            
            f.write(f"## Overall Results\n\n")
            f.write(f"- Total Experiments: {len(self.results)}\n")
            f.write(f"- Successful: {len(df_success)} ({len(df_success)/len(self.results)*100:.1f}%)\n")
            f.write(f"- Failed: {len(df) - len(df_success)}\n\n")
            
            if not df_success.empty and 'test_accuracy' in df_success.columns:
                f.write(f"## Accuracy Summary\n\n")
                summary = df_success.groupby('architecture')['test_accuracy'].agg(['mean', 'std', 'min', 'max'])
                f.write(summary.to_markdown())
                f.write("\n\n")
                
                f.write(f"## Best Results per Dataset\n\n")
                for dataset in df_success['dataset'].unique():
                    best = df_success[df_success['dataset'] == dataset].nlargest(1, 'test_accuracy').iloc[0]
                    f.write(f"**{dataset}:** {best['architecture']} + {best['backbone']} = {best['test_accuracy']:.4f}\n")
                f.write("\n")
            
            f.write(f"## Files Generated\n\n")
            f.write(f"- Results CSV: `results.csv`\n")
            f.write(f"- Results JSON: `results.json`\n")
            f.write(f"- Visualizations: `plots/`\n")
            f.write(f"- Logs: `logs/`\n")
        
        print(f"📄 Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete benchmark of 5 architectures × 4 datasets × 4 backbones"
    )
    
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per experiment')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-qubits', type=int, default=4)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--shots', type=int, default=None, help='Shots for Qiskit (None=ideal)')
    parser.add_argument('--noise-1q', type=float, default=0.001)
    parser.add_argument('--noise-2q', type=float, default=0.01)
    
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=DATASETS + ['all'],
                        help='Datasets to run (default: all)')
    parser.add_argument('--architectures', nargs='+', default=None,
                        choices=ARCHITECTURES + ['all'],
                        help='Architectures to run (default: all)')
    parser.add_argument('--backbones', nargs='+', default=None,
                        choices=BACKBONES + ['all'],
                        help='Backbones to run (default: all)')
    
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test: 1 epoch, only hymenoptera + resnet18')
    
    args = parser.parse_args()
    
    # Quick test override
    if args.quick_test:
        args.epochs = 1
        args.datasets = ['hymenoptera']
        args.backbones = ['resnet18']
        print("[INFO] Quick-test mode enabled: 1 epoch, hymenoptera, resnet18")
    
    # Process 'all'
    datasets = DATASETS if not args.datasets or 'all' in args.datasets else args.datasets
    architectures = ARCHITECTURES if not args.architectures or 'all' in args.architectures else args.architectures
    backbones = BACKBONES if not args.backbones or 'all' in args.backbones else args.backbones
    
    # Create runner
    runner = BenchmarkRunner(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_qubits=args.n_qubits,
        depth=args.depth,
        learning_rate=args.lr,
        gamma=args.gamma,
        shots=args.shots,
        noise_1q=args.noise_1q,
        noise_2q=args.noise_2q
    )
    
    # Run benchmark
    runner.run_all(datasets=datasets, architectures=architectures, backbones=backbones)
    
    # Generate visualizations and report
    runner.generate_visualizations()
    runner.generate_summary_report()
    
    print("\n✅ Benchmark completed successfully!")


if __name__ == '__main__':
    main()
