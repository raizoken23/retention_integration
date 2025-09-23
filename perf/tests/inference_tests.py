import torch
import pytest
from functools import partial
from retention.create_inputs import create_inference_inputs, inference_input_properties, inference_output_properties
from retention.triton import power_retention_inference as power_retention_triton_inference
from retention.reference import power_retention_inference as power_retention_inference_ref
from perf._checks import diff
from perf.tests.test_list import fn_set_and_param_range_to_test_cases

inference_input_output_triton = {'create_inputs': partial(create_inference_inputs, fused_normalizer=False), 'input_properties': partial(inference_input_properties, fused_normalizer=False), 'output_properties': partial(inference_output_properties, fused_normalizer=False)}

inference_param_rangs = {
    'b': [1],
    'tq': [1], 
    'tk': [129],
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [2, 7],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False],
    'chunk_size': [None, 128],
    'deg': [2],
    'initial_state': [True, False],
    'switch_over_seq_len': [129],
}

inference_fn_sets = [
    {'name': 'power_retention_triton_inference_state_parity', 'extends': 'power_retention', 'impl': 'triton',
        'fn': power_retention_triton_inference, **inference_input_output_triton},
    {'name': 'power_retention_triton_inference_state_parity_ref', 'extends': 'power_retention', 'impl': 'ref',
        'fn': power_retention_inference_ref, **inference_input_output_triton},
]

TEST_CASES = fn_set_and_param_range_to_test_cases(inference_fn_sets, inference_param_rangs)


@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_inference_state_parity(fns_params):
    fns, params = fns_params
    inputs = fns['create_inputs'](**params)
    
    output, state, sum_of_keys = fns['fn'](**inputs)

    new_inputs = {**inputs, 'initial_state': state, 'sum_of_keys': sum_of_keys, 'K': None, 'V': None, 'log_G': None}

    new_output, new_state, new_sum_of_keys = fns['fn'](**new_inputs)

    diff(state, new_state, atol=1e-2, rtol=1e-3)
    diff(sum_of_keys, new_sum_of_keys, atol=1e-2, rtol=1e-3)
    diff(output, new_output, atol=1e-2, rtol=1e-3)