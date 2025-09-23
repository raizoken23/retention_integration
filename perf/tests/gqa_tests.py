import torch
import pytest
from functools import partial
from retention import input_properties, inference_input_properties, output_properties, inference_output_properties
from retention.reference import power_retention as power_retention_reference, power_retention_inference as power_retention_reference_inference
from retention.triton import power_retention as power_retention_triton, power_retention_inference as power_retention_triton_inference
from retention.vidrial import power_retention as power_retention_vidrial, power_retention_inference as power_retention_vidrial_inference
from retention.vidrial_reference import power_retention as power_retention_vidrial_reference, power_retention_inference as power_retention_vidrial_reference_inference
from retention.create_inputs import create_inputs, create_inference_inputs
from perf.tests.test_list import fn_set_and_param_range_to_test_cases
from perf._checks import check_fn_forwards_match, check_fn_backwards_match

# common inputs and output stubs
input_output = {'create_inputs': create_inputs, 'input_properties': input_properties, 'output_properties': output_properties}
inference_input_output_vidrial = {'create_inputs': partial(create_inference_inputs, fused_normalizer=True), 'input_properties': partial(inference_input_properties, fused_normalizer=True), 'output_properties': partial(inference_output_properties, fused_normalizer=True)}
inference_input_output_triton = {'create_inputs': partial(create_inference_inputs, fused_normalizer=False), 'input_properties': partial(inference_input_properties, fused_normalizer=False), 'output_properties': partial(inference_output_properties, fused_normalizer=False)}

gqa_param_ranges = {
    'b': [1],
    't': [32, 1024], 
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [2, 7],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 32],
    'deg': [2],
}

gqa_inference_param_ranges = {
    'b': [1],
    'tq': [128, 1024], 
    'tk': [129, 256],
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [2, 7],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 128],
    'deg': [2],
    'initial_state': [True, False],
}

gqa_fn_sets = [
    {'name': 'power_retention_vidrial_gqa_reference', 'extends': 'power_retention', 'impl': 'reference',
        'fn': power_retention_vidrial_reference, **input_output},
    {'name': 'power_retention_vidrial_gqa', 'extends': 'power_retention', 'impl': 'vidrial',
        'fn': power_retention_vidrial, **input_output},
    {'name': 'power_retention_triton_gqa_reference', 'extends': 'power_retention', 'impl': 'triton',
        'fn': power_retention_reference, **input_output},
    {'name': 'power_retention_triton_gqa', 'extends': 'power_retention', 'impl': 'triton',
        'fn': power_retention_triton, **input_output},
]

gqa_inference_fn_sets = [
    {'name': 'power_retention_vidrial_gqa_inference_reference', 'extends': 'power_retention', 'impl': 'reference',
        'fn': power_retention_vidrial_reference_inference, **inference_input_output_vidrial},
    {'name': 'power_retention_vidrial_gqa_inference', 'extends': 'power_retention', 'impl': 'vidrial',
        'fn': power_retention_vidrial_inference, **inference_input_output_vidrial},
    {'name': 'power_retention_triton_gqa_inference_reference', 'extends': 'power_retention', 'impl': 'triton',
        'fn': power_retention_reference_inference, **inference_input_output_triton},
    {'name': 'power_retention_triton_gqa_inference', 'extends': 'power_retention', 'impl': 'triton',  
        'fn': power_retention_triton_inference, **inference_input_output_triton},
]

TEST_CASES = fn_set_and_param_range_to_test_cases(gqa_fn_sets, gqa_param_ranges)
FWD_TEST_CASES = TEST_CASES + fn_set_and_param_range_to_test_cases(gqa_inference_fn_sets, gqa_inference_param_ranges) # include inference as well for fwd tests


def gqa_wrapper(fn):
    def slice_outputs(outputs, qhead_ratio):
        return (outputs[0], *[outputs[i][:, ::qhead_ratio] if outputs[i] is not None else None for i in range(1, len(outputs))])

    def wrapped(**inputs):
        res = inputs.copy()
        qhead_ratio = inputs['Q'].shape[2] // inputs['K'].shape[2]
        assert qhead_ratio >= 1, 'qhead_ratio must be greater than or equal to 1'
        res['K'] = inputs['K'].repeat_interleave(qhead_ratio, dim=2)
        res['V'] = inputs['V'].repeat_interleave(qhead_ratio, dim=2)
        if inputs['log_G'] is not None:
            res['log_G'] = inputs['log_G'].repeat_interleave(qhead_ratio, dim=2)
        if inputs['initial_state'] is not None:
            res['initial_state'] = inputs['initial_state'].repeat_interleave(qhead_ratio, dim=1)
        if 'sum_of_keys' in inputs and inputs['sum_of_keys'] is not None:
            res['sum_of_keys']  = inputs['sum_of_keys'].repeat_interleave(qhead_ratio, dim=1)

        outputs = fn(**res)
        if isinstance(outputs, tuple): # state needs to be sliced to the original shape
            outputs = slice_outputs(outputs, qhead_ratio)
        return outputs
    
    return wrapped

@pytest.mark.parametrize("fns_params", FWD_TEST_CASES)
def test_gqa_fwd(fns_params):
    fns, params = fns_params
    if 'tq' in params and 'tk' in params and params['tq'] > params['tk']:
        pytest.skip('tq must be <= tk')
    inputs = fns['create_inputs'](**params)

    check_fn_forwards_match(
        ref_fn=gqa_wrapper(fns['fn']),
        gold_inputs=inputs,
        test_fn=fns['fn'],
        test_inputs=inputs,
        rtol=1e-2,
        atol=5e-3,
    )


@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_gqa_bwd(fns_params):
    fns, params = fns_params
    if 'tq' in params and 'tk' in params and params['tq'] > params['tk']:
        pytest.skip('tq must be <= tk')
    gold_inputs = fns['create_inputs'](**(params | {'requires_grad': True, 'dtype': torch.float32}))
    test_inputs = fns['create_inputs'](**(params | {'requires_grad': True}))

    check_fn_backwards_match(
        ref_fn=gqa_wrapper(fns['fn']),
        gold_inputs=gold_inputs,
        test_fn=fns['fn'],
        test_inputs=test_inputs,
        atol=5e-2,
        rtol=2e-2,
        diff_tol=0.025,
    )