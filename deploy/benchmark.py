import subprocess
import itertools
import tempfile
import sys
from tqdm import tqdm
import logging
import json

# CASES = {
#     'batch_sizes': [2],
#     'prefill_contexts': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
#     'generation_contexts': [128, 512, 1024],
#     'chunk_sizes': [128, 512, 1024, 2048],
#     'switch_over_seq_lens': [128, 512, 1024, 2048, 4096, 8192],
# }

CASES = {
    'batch_sizes': [2],
    'prefill_contexts': [1024],
    'generation_contexts': [128],
    'chunk_sizes': [128],
    'switch_over_seq_lens': [512],
}


def skip_cases(case):
    if case['prefill_contexts'] >= 16384 and case['generation_contexts'] >= 128:
        return True
    return False

def benchmark_generation():
    cases = list({k: v for k, v in zip(CASES.keys(), case)} for case in itertools.product(*CASES.values()))
    cases = [case for case in cases if not skip_cases(case)]
    python_path = sys.executable
    res = []
    for case in tqdm(cases):
        with tempfile.NamedTemporaryFile(suffix='.json') as f:
            subprocess.run([
                python_path, 'generate.py',
                '--batch-size', str(case['batch_sizes']),
                '--prefill-context', str(case['prefill_contexts']),
                '--max-new-tokens', str(case['generation_contexts']), 
                '--chunk-size', str(case['chunk_sizes']),
                '--switch-over-seq-len', str(case['switch_over_seq_lens']),
                '--stats-only', '--stats-file', f.name,
                '--model', 'manifestai/powercoder-3b',
                '--label', 'powercoder'
                '--no-stream',
            ])
            subprocess.run([
                python_path, 'generate.py',
                '--batch-size', str(case['batch_sizes']),
                '--prefill-context', str(case['prefill_contexts']),
                '--max-new-tokens', str(case['generation_contexts']), 
                '--chunk-size', str(case['chunk_sizes']),
                '--switch-over-seq-len', str(case['switch_over_seq_lens']),
                '--stats-only', '--stats-file', f.name,
                '--model', 'bigcode/starcoder2-3b',
                '--label', 'transformer',
                '--disable-sliding-window',
                '--no-stream',
            ])
            with open(f.name, 'r') as f:
                stats = json.load(f)
            res.append({
                'case': case,
                'stats': stats,
            })
            
    print(res)
    import pdb; pdb.set_trace()
            

if __name__ == '__main__':
    benchmark_generation()