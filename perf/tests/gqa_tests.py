import torch
import pytest
from power_attention import input_properties, inference_input_properties, output_properties, inference_output_properties
from power_attention.vidrial import power_full, power_full_inference
from power_attention.vidrial_reference import power_full as power_full_reference, power_full_inference as power_full_reference_inference
from power_attention.create_inputs import create_vidrial_inputs, create_inference_inputs
from perf.tests.test_list import fn_set_and_param_range_to_test_cases
from perf._checks import check_fn_forwards_match, check_fn_backwards_match
input_output = {'create_inputs': create_vidrial_inputs, 'input_properties': input_properties, 'output_properties': output_properties}
inference_input_output = {'create_inputs': create_inference_inputs, 'input_properties': inference_input_properties, 'output_properties': inference_output_properties}

gqa_param_ranges = {
    'b': [1],
    't': [32, 1024], 
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [2],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 32],
    'deg': [2],
}

gqa_inference_param_ranges = {
    'b': [1],
    't': [128], 
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [2],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 128],
    'deg': [2],
    'initial_state': [True, False],
}

gqa_fn_sets = [
    {'name': 'power_full_vidrial_gqa_reference', 'extends': 'power_full', 'impl': 'reference',
        'fn': power_full_reference, **input_output},
    {'name': 'power_full_vidrial_gqa', 'extends': 'power_full', 'impl': 'vidrial',
        'fn': power_full, **input_output},
]

gqa_inference_fn_sets = [
    {'name': 'power_full_vidrial_gqa_inference_reference', 'extends': 'power_full', 'impl': 'reference',
        'fn': power_full_reference_inference, **inference_input_output},
    {'name': 'power_full_vidrial_gqa_inference', 'extends': 'power_full', 'impl': 'vidrial',
        'fn': power_full_inference, **inference_input_output},
]

TEST_CASES = fn_set_and_param_range_to_test_cases(gqa_fn_sets, gqa_param_ranges)
FWD_TEST_CASES = TEST_CASES + fn_set_and_param_range_to_test_cases(gqa_inference_fn_sets, gqa_inference_param_ranges) # include inference as well for fwd tests


def gqa_wrapper(fn):
    def wrapped(**inputs):
        qhead_ratio = inputs['Q'].shape[2] // inputs['K'].shape[2]
        assert qhead_ratio >= 1, 'qhead_ratio must be greater than or equal to 1'
        K = inputs['K'].repeat_interleave(qhead_ratio, dim=2)
        V = inputs['V'].repeat_interleave(qhead_ratio, dim=2)
        if inputs['log_G'] is not None:
            log_G = inputs['log_G'].repeat_interleave(qhead_ratio, dim=2)
        else:
            log_G = None
        if inputs['initial_state'] is not None:
            initial_state = inputs['initial_state'].repeat_interleave(qhead_ratio, dim=1)
        else:
            initial_state = None
        res = inputs.copy()
        res['K'] = K
        res['V'] = V
        if log_G is not None:
            res['log_G'] = log_G
        if initial_state is not None:
            res['initial_state'] = initial_state
        outputs = fn(**res)
        if isinstance(outputs, tuple): # state needs to be sliced to the original shape
            outputs = (outputs[0], outputs[1][:, ::qhead_ratio, :, :] if outputs[1] is not None else None)
        return outputs
    
    return wrapped

@pytest.mark.parametrize("fns_params", FWD_TEST_CASES)
def test_gqa_fwd(fns_params):
    fns, params = fns_params
    inputs = fns['create_inputs'](**params)

    check_fn_forwards_match(
        ref_fn=gqa_wrapper(fns['fn']),
        gold_inputs=inputs,
        test_fn=fns['fn'],
        test_inputs=inputs,
        rtol=1e-2,
        atol=1e-3,
    )


@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_gqa_bwd(fns_params):
    fns, params = fns_params
    gold_inputs = fns['create_inputs'](**(params | {'requires_grad': True, 'dtype': torch.float32}))
    test_inputs = fns['create_inputs'](**(params | {'requires_grad': True}))

    check_fn_backwards_match(
        ref_fn=gqa_wrapper(fns['fn']),
        gold_inputs=gold_inputs,
        test_fn=fns['fn'],
        test_inputs=test_inputs,
        atol=2e-2,
        rtol=1e-2,
        diff_tol=0.025,
    )