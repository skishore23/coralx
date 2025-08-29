"""
Baseline Performance Evaluation Service
Integrated into CORAL-X framework for automatic baseline vs LoRA comparison
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataclasses import dataclass

from ..common.logging import get_logger
from ..common.config import BaselineTestConfig


@dataclass
class BaselineResults:
    """Results from baseline model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fake_detection_accuracy: float
    real_detection_accuracy: float
    avg_confidence: float
    total_samples: int


@dataclass
class ComparisonResults:
    """Results comparing LoRA adapter to baseline."""
    baseline: BaselineResults
    lora: BaselineResults
    improvement: Dict[str, float]
    verdict: str
    is_significant_improvement: bool


class BaselineEvaluator:
    """Service for evaluating baseline vs LoRA adapter performance."""
    
    def __init__(self, config: BaselineTestConfig, dataset_path: str):
        self.config = config
        self.dataset_path = dataset_path
        self.logger = get_logger(self.__class__.__name__)
        
        # Cache test dataset to avoid reloading
        self._test_data = None
    
    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load challenging test dataset for evaluation."""
        
        if self._test_data is not None:
            return self._test_data
        
        self.logger.info(f"Loading test dataset from {self.dataset_path}")
        
        # Load actual dataset files
        fake_file = Path(self.dataset_path) / "PolitiFact_fake_news_content.csv"
        real_file = Path(self.dataset_path) / "PolitiFact_real_news_content.csv"
        
        if not fake_file.exists() or not real_file.exists():
            raise FileNotFoundError(f"Dataset files not found in {self.dataset_path}")
        
        fake_df = pd.read_csv(fake_file)
        real_df = pd.read_csv(real_file)
        
        # Create challenging test set
        test_data = []
        samples_per_class = self.config.test_samples // 2
        
        # Sample diverse fake news
        fake_sample = fake_df.sample(n=samples_per_class, random_state=42)
        for _, row in fake_sample.iterrows():
            text = self._extract_text(row)
            if text and len(text) > 30:  # Filter short texts
                test_data.append({
                    'text': text,
                    'label': 1,  # fake
                    'source': 'fake_news'
                })
        
        # Sample diverse real news
        real_sample = real_df.sample(n=samples_per_class, random_state=42)
        for _, row in real_sample.iterrows():
            text = self._extract_text(row)
            if text and len(text) > 30:
                test_data.append({
                    'text': text,
                    'label': 0,  # real
                    'source': 'real_news'
                })
        
        # Shuffle dataset
        np.random.seed(42)
        np.random.shuffle(test_data)
        
        self._test_data = test_data
        self.logger.info(f"Loaded {len(test_data)} test samples ({len([x for x in test_data if x['label']==1])} fake, {len([x for x in test_data if x['label']==0])} real)")
        
        return test_data
    
    def _extract_text(self, row: pd.Series) -> Optional[str]:
        """Extract text from dataset row, preferring longer content."""
        
        # Try different text fields in order of preference
        for field in ['text', 'title', 'content', 'news_body']:
            if field in row and pd.notna(row[field]):
                text = str(row[field])[:300]  # Limit length
                if len(text.strip()) > 10:
                    return text.strip()
        
        return None
    
    def evaluate_model(self, model, tokenizer, model_name: str = "Model") -> BaselineResults:
        """Evaluate a model with comprehensive metrics."""
        
        self.logger.info(f"Evaluating {model_name} performance...")
        
        test_data = self._load_test_dataset()
        device = next(model.parameters()).device
        
        predictions = []
        confidence_scores = []
        true_labels = [item['label'] for item in test_data]
        
        for i, item in enumerate(test_data):
            text = item['text']
            true_label = item['label']
            
            # Use multiple prompt styles for robustness
            prompt_template = self.config.prompt_styles[i % len(self.config.prompt_styles)]
            prompt = prompt_template.format(text=text)
            
            # Multiple attempts for confidence estimation
            responses = []
            for _ in range(self.config.multiple_attempts):
                # Tokenize and generate
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = response[len(prompt):].strip().lower()
                responses.append(generated)
            
            # Analyze response consistency
            fake_votes = sum(1 for r in responses if 'fake' in r)
            real_votes = sum(1 for r in responses if 'real' in r)
            
            if fake_votes > real_votes:
                pred = 1
                confidence = fake_votes / len(responses)
            elif real_votes > fake_votes:
                pred = 0
                confidence = real_votes / len(responses)
            else:
                pred = 0  # Default to real when uncertain
                confidence = 0.33  # Low confidence
            
            predictions.append(pred)
            confidence_scores.append(confidence)
            
            # Log first few examples for debugging
            if i < 3:
                self.logger.debug(f"Example {i+1}: True={true_label}, Pred={pred}, Conf={confidence:.2f}, Responses={responses}")
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0)
        avg_confidence = np.mean(confidence_scores)
        
        # Calculate per-class accuracy
        fake_indices = [i for i, label in enumerate(true_labels) if label == 1]
        real_indices = [i for i, label in enumerate(true_labels) if label == 0]
        
        fake_accuracy = accuracy_score([true_labels[i] for i in fake_indices], 
                                       [predictions[i] for i in fake_indices]) if fake_indices else 0
        real_accuracy = accuracy_score([true_labels[i] for i in real_indices], 
                                       [predictions[i] for i in real_indices]) if real_indices else 0
        
        results = BaselineResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            fake_detection_accuracy=fake_accuracy,
            real_detection_accuracy=real_accuracy,
            avg_confidence=avg_confidence,
            total_samples=len(test_data)
        )
        
        self.logger.info(f"{model_name} Results: Accuracy={accuracy:.1%}, F1={f1:.3f}, FakeDetection={fake_accuracy:.1%}")
        
        return results
    
    def compare_baseline_vs_lora(self, base_model, tokenizer, lora_adapter_path: str) -> ComparisonResults:
        """Compare baseline model vs LoRA adapter performance."""
        
        self.logger.info(f"Comparing baseline vs LoRA adapter: {lora_adapter_path}")
        
        # Evaluate baseline
        baseline_results = self.evaluate_model(base_model, tokenizer, "BASELINE")
        
        # Load and evaluate LoRA model
        try:
            from peft import PeftModel
            lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            lora_results = self.evaluate_model(lora_model, tokenizer, f"LORA")
        except Exception as e:
            self.logger.error(f"Failed to load LoRA adapter: {e}")
            raise
        
        # Calculate improvements
        improvement = {
            'accuracy': lora_results.accuracy - baseline_results.accuracy,
            'f1_score': lora_results.f1_score - baseline_results.f1_score,
            'fake_detection': lora_results.fake_detection_accuracy - baseline_results.fake_detection_accuracy,
            'confidence': lora_results.avg_confidence - baseline_results.avg_confidence
        }
        
        # Determine verdict
        accuracy_improvement = improvement['accuracy']
        
        if accuracy_improvement > 0.15:  # 15%+ improvement
            verdict = "EXCELLENT: Major improvement detected"
            is_significant = True
        elif accuracy_improvement > 0.10:  # 10%+ improvement
            verdict = "VERY GOOD: Significant improvement detected"
            is_significant = True
        elif accuracy_improvement > self.config.improvement_threshold:  # Above threshold
            verdict = "GOOD: Clear improvement detected"
            is_significant = True
        elif accuracy_improvement > 0.01:  # 1%+ improvement
            verdict = "MODEST: Slight improvement detected"
            is_significant = False
        else:
            verdict = "NO IMPROVEMENT: LoRA performs similarly to baseline"
            is_significant = False
        
        comparison = ComparisonResults(
            baseline=baseline_results,
            lora=lora_results,
            improvement=improvement,
            verdict=verdict,
            is_significant_improvement=is_significant
        )
        
        self.logger.info(f"Comparison verdict: {verdict} (Accuracy: {accuracy_improvement:+.1%})")
        
        return comparison
    
    def print_detailed_results(self, comparison: ComparisonResults) -> None:
        """Print detailed comparison results."""
        
        print(f"\n" + "="*60)
        print(f"📊 BASELINE vs LORA PERFORMANCE COMPARISON")
        print(f"="*60)
        
        print(f"\n📊 BASELINE PERFORMANCE:")
        baseline = comparison.baseline
        print(f"   Overall Accuracy: {baseline.accuracy:.3f} ({baseline.accuracy*100:.1f}%)")
        print(f"   Fake Detection: {baseline.fake_detection_accuracy:.3f} ({baseline.fake_detection_accuracy*100:.1f}%)")
        print(f"   Real Detection: {baseline.real_detection_accuracy:.3f} ({baseline.real_detection_accuracy*100:.1f}%)")
        print(f"   F1-Score: {baseline.f1_score:.3f}")
        print(f"   Average Confidence: {baseline.avg_confidence:.3f}")
        
        print(f"\n🚀 LORA ADAPTER PERFORMANCE:")
        lora = comparison.lora
        print(f"   Overall Accuracy: {lora.accuracy:.3f} ({lora.accuracy*100:.1f}%)")
        print(f"   Fake Detection: {lora.fake_detection_accuracy:.3f} ({lora.fake_detection_accuracy*100:.1f}%)")
        print(f"   Real Detection: {lora.real_detection_accuracy:.3f} ({lora.real_detection_accuracy*100:.1f}%)")
        print(f"   F1-Score: {lora.f1_score:.3f}")
        print(f"   Average Confidence: {lora.avg_confidence:.3f}")
        
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        for metric, improvement in comparison.improvement.items():
            print(f"   {metric.replace('_', ' ').title()}: {improvement:+.3f} ({improvement*100:+.1f}pp)")
        
        print(f"\n🎯 VERDICT: {comparison.verdict}")
        
        if comparison.is_significant_improvement:
            print(f"✅ SUCCESS: Your evolved LoRA adapter shows meaningful improvement!")
            print(f"💪 The evolutionary process is working effectively")
        else:
            print(f"⚠️  LIMITED SUCCESS: Improvement is minimal")
            print(f"💡 Consider: more training data, longer training, or different LoRA configuration")
        
        print(f"\n📝 Test Details:")
        print(f"   Test Samples: {baseline.total_samples}")
        print(f"   Multiple Attempts: {self.config.multiple_attempts}")
        print(f"   Improvement Threshold: {self.config.improvement_threshold*100:.1f}%")