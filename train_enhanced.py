"""
Standalone Face Recognition Training & Accuracy Evaluation Script
Trains on augmented facial data, evaluates metrics, and saves the enhanced model.
"""

import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine.ml_engine import FaceRecognitionPipeline


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Face Recognition Model with Augmented Data")
    parser.add_argument("--synthetic", action="store_true", help="Add diverse benchmark synthetic profiles for multi-class testing")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"), help="Path to data directory")
    args = parser.parse_args()

    print("=" * 65)
    print("   AURA NEURAL FACE RECOGNITION | ENHANCED TRAINING STUDIO")
    print("=" * 65)

    pipeline = FaceRecognitionPipeline(data_dir=args.data_dir)

    raw_faces, raw_names = pipeline.load_raw_dataset()
    print(f"[*] Raw Dataset Loaded: {len(raw_names)} samples across {len(set(raw_names))} identities.")

    if args.synthetic or len(set(raw_names)) < 2:
        print("[+] Synthesizing diverse benchmark profiles for multi-class evaluation...")
        added = pipeline.generate_synthetic_benchmark_profiles()
        print(f"[+] Added {added} reference benchmark samples.")

    print("\n[*] Starting Multi-Condition Data Augmentation:")
    print("    - Horizontal mirror reflection")
    print("    - Rotations (-12deg, -6deg, +6deg, +12deg)")
    print("    - CLAHE illumination normalization")
    print("    - Gamma correction & brightness shifts")
    print("    - Scale/zoom & subtle Gaussian blur")

    print("\n[*] Extracting Spatial, Gradient & Texture Features...")
    print("[*] Performing PCA Orthogonal Space Projection...")
    print("[*] Training Calibrated RBF-SVM + Cosine-KNN Ensemble...")

    t0 = time.time()
    metrics = pipeline.train_enhanced_pipeline()
    elapsed = round(time.time() - t0, 2)

    if not metrics.get("success"):
        print(f"[!] Training failed: {metrics.get('error')}")
        return

    print("\n" + "=" * 65)
    print("                     TRAINING RESULTS & ACCURACY")
    print("=" * 65)
    print(f"Total Original Samples:   {metrics['original_samples']}")
    print(f"Total Augmented Samples:  {metrics['augmented_samples']} ({metrics['expansion_ratio']} expansion)")
    print(f"Enrolled Students ({metrics['student_count']}):    {', '.join(metrics['enrolled_students'])}")
    print(f"PCA Variance Retained:    {metrics['pca_variance_explained']}% in {metrics['pca_components']} components")
    print(f"Training Time:            {elapsed}s")
    print("-" * 65)
    print(f"Enhanced Model Accuracy:  {metrics['enhanced_model']['accuracy']}% (Precision: {metrics['enhanced_model']['precision']}%, F1: {metrics['enhanced_model']['f1_score']}%)")
    print(f"Baseline KNN Accuracy:    {metrics['baseline_model']['accuracy']}%")
    print(f"Accuracy Gain:            +{metrics['improvement']['accuracy_gain_pct']}%")
    print(f"False Positive Reduction: {metrics['improvement']['false_positive_reduction']}")
    print(f"Lighting Invariance:      {metrics['enhanced_model']['lighting_invariance']}")
    print(f"Unknown Rejection:        {metrics['enhanced_model']['unknown_rejection']}")
    print("=" * 65)
    print("[SUCCESS] Model saved to data/enhanced_model.pkl")
    print("[SUCCESS] Benchmark report saved to data/benchmark_metrics.json\n")


if __name__ == "__main__":
    main()
