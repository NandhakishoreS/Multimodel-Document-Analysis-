import sys, json, time
from pathlib import Path
PROJECT_ROOT = Path(r'c:\MDA\multimodal-document-ai')
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / 'models/checkpoints/layoutlmv3/sroie/full'
META_PATH = CHECKPOINT_DIR / 'checkpoint_meta.json'
with open(META_PATH) as f:
    meta = json.load(f)

training_summary = {
    'checkpoint_dir': str(CHECKPOINT_DIR),
    'best_val_macro_f1': meta['val_macro_f1'],
    'epochs_trained': meta['epoch'],
    'training_time_s': 2871.0,
    'hardware': {'gpu_name': 'NVIDIA GeForce RTX 3060 Laptop GPU', 'vram_gb': 6.44, 'cuda_version': '12.1'},
    'config': {'lr': 5e-5, 'batch_size': 4, 'gradient_accumulation_steps': 2, 'effective_batch_size': 8, 'epochs': 10, 'fp16': True},
}

print('Loading evaluation pipeline...')
from evaluation.evaluate_layoutlmv3 import evaluate_dataset
print('Running test evaluation on 361 docs...')
t0 = time.time()
eval_res = evaluate_dataset(dataset_name='sroie', split='test', checkpoint_dir=str(CHECKPOINT_DIR))
print(f'Done in {time.time()-t0:.1f}s')
print(json.dumps(eval_res, indent=2))
with open(PROJECT_ROOT / 'reports/sroie_layoutlmv3_results.json', 'w') as f:
    json.dump({'dataset':'sroie','model':'microsoft/layoutlmv3-base','hardware':training_summary['hardware'],'training_config':training_summary['config'],'best_val_macro_f1':training_summary['best_val_macro_f1'],'epochs_trained':training_summary['epochs_trained'],'training_time_s':training_summary['training_time_s'],'test_evaluation':eval_res}, f, indent=2)
print('JSON saved.')
