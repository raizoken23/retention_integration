import torch
from copy import deepcopy

def check_tensor(tensor, properties=None):
    """Check that a single tensor is well-behaved and optionally matches expected properties.

    Args:
        tensor (torch.Tensor): The tensor to check
        properties (tuple, dict, or object, optional): If provided, checks that tensor matches these properties:
            - A tuple/list of (shape, dtype, device)
            - A dict with 'shape', 'dtype', and 'device' keys
            - An object with shape, dtype, and device attributes

    Raises:
        AssertionError: If tensor is not well-behaved or does not match the expected properties
    """
    # Handle tuple unpacking
    if properties is None:
        pass
    elif isinstance(properties, (list, tuple)):
        shape, dtype, device = properties
    # Handle dict
    elif isinstance(properties, dict):
        shape = properties['shape']
        dtype = properties['dtype'] 
        device = properties['device']
    # Handle object with attributes
    else:
        shape = properties.shape
        dtype = properties.dtype
        device = properties.device
    if properties is not None:
        assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
        assert same_device(tensor.device, torch.device(device)), f"Expected device {device}, got {tensor.device}"
    if not torch.isfinite(tensor).all():
        non_finite_indices = torch.argwhere(~torch.isfinite(tensor))
        torch.set_printoptions(edgeitems=10)
        raise AssertionError(f"Tensor contains non-finite values at indices: {non_finite_indices}")
    # assert tensor.is_contiguous(), "Tensor must be contiguous"
    # assert torch.all(torch.abs(tensor) < 1e6), "Tensor contains values with magnitude >= 1e6"

def same_device(device1, device2):
    # Check if types match
    if device1.type != device2.type:
        return False
    # Check if indices match or if either index is None (meaning default GPU)
    return (device1.index == device2.index) or (device1.index is None or device2.index is None)


def check_tensors_properties(tensors_or_tensor, properties_or_property):
    """Check that tensor(s) match expected properties.
    
    Args:
        tensors_or_tensor: Single tensor or list of tensors to check
        properties_or_property: Properties to check against, specified as either:
            For a single tensor:
                - A tuple/list of (shape, dtype, device)
                - A dict with 'shape', 'dtype', and 'device' keys
                - An object with shape, dtype, and device attributes
                - None to skip checking properties
            For multiple tensors:
                - List of properties in above formats, same length as tensors
    """
    if isinstance(tensors_or_tensor, torch.Tensor):
        check_tensor(tensors_or_tensor, properties_or_property)
    else:
        if len(tensors_or_tensor) != len(properties_or_property):
            raise ValueError("Number of tensors must match number of properties")
        for i, (tensor, props) in enumerate(zip(tensors_or_tensor, properties_or_property)):
            try:
                check_tensor(tensor, props)
            except AssertionError as e:
                raise AssertionError(f"Element {i}/{len(tensors_or_tensor)} failed: {e}")

def check_tensor_property_pairs(*tensor_prop_pairs):
    """Check that tensors match properties specified as pairs.
    
    Args:
        tensor_props_pairs (iterable): Iterable of (tensor, property) pairs
    """
    for tensor, property in tensor_prop_pairs:
        check_tensor(tensor, property)

def save_to_csv(tensor, filename):
    """
    Save a tensor to a CSV file.
    """
    import pandas as pd
    t = tensor.squeeze().detach().cpu().to(torch.float32).numpy()
    df = pd.DataFrame(t)
    df.to_csv(filename, index=False, header=False)
    print(f"Tensor saved to {filename}")

def check_all_equivalent(a, b, rtol=1e-3, atol=1e-3, ignore_nan=False):
    """Check if two objects or iterables of objects are equivalent.
    
    Args:
        a: First object or iterable of objects
        b: Second object or iterable of objects to compare against
        rtol: Relative tolerance for allclose check (for tensors)
        atol: Absolute tolerance for allclose check (for tensors)
    """
    # Handle single objects
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        assert a.shape == b.shape, f"Shapes don't match: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"Dtypes don't match: {a.dtype} vs {b.dtype}" 
        assert a.device == b.device, f"Devices don't match: {a.device} vs {b.device}"
        assert a.is_contiguous() == b.is_contiguous(), "Contiguity doesn't match"
        has_nans = (torch.isnan(a) | torch.isnan(b)).any()
        if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):    
            if has_nans and not ignore_nan:
                raise AssertionError(f"One or both inputs contain nans but otherwise match")
        else:
            diff = torch.abs(a - b)
            abs_tol = atol
            rel_tol = rtol * torch.abs(b)
            
            # Find failures of absolute threshold
            abs_failures = diff > abs_tol
            if abs_failures.any():
                rel_error_for_abs_fails = (diff[abs_failures] / (torch.abs(b[abs_failures]) + 1e-8))
                max_rel_error = rel_error_for_abs_fails.max().item()
                max_rel_idx = torch.argwhere(diff / (torch.abs(b) + 1e-8) == max_rel_error)[0]
            
            # Find failures of relative threshold  
            rel_failures = diff > rel_tol
            if rel_failures.any():
                abs_error_for_rel_fails = diff[rel_failures]
                max_abs_error = abs_error_for_rel_fails.max().item()
                max_abs_idx = torch.argwhere(diff == max_abs_error)[0]

            msg = f"Values don't match within tolerance rtol={rtol}, atol={atol}. "
            msg += f"Found {(abs_failures | rel_failures).sum().item()}/{a.numel()} failing indices. "
            
            if abs_failures.any():
                msg += f"Maximum relative error among absolute failures: {max_rel_error} at index {tuple(max_rel_idx.tolist())} "
                msg += f"(a={a[tuple(max_rel_idx)].item():.6f}, b={b[tuple(max_rel_idx)].item():.6f}). "
            
            if rel_failures.any():
                msg += f"Maximum absolute error among relative failures: {max_abs_error} at index {tuple(max_abs_idx.tolist())} "
                msg += f"(a={a[tuple(max_abs_idx)].item():.6f}, b={b[tuple(max_abs_idx)].item():.6f})"

            if has_nans:
                msg += f". Also, one or both inputs contain NaN values."
            raise AssertionError(msg)
        return
    elif a is None and b is None:
        return
    elif isinstance(a, (float, int, bool, str)) and isinstance(b, (float, int, bool, str)):
        assert a == b, f"Values don't match: {a} vs {b}"
        return
        
    # Handle iterables of objects
    try:
        for i, (a_item, b_item) in enumerate(zip(a, b)):
            try:
                check_all_equivalent(a_item, b_item, rtol=rtol, atol=atol, ignore_nan=ignore_nan)
            except AssertionError as e:
                raise AssertionError(f"Element {i}/{len(a)} failed: {e}")
    except TypeError as e:
        if "zip" not in str(e):
            raise
        raise TypeError(f"Inputs must both be tensors, scalars, or both be iterables of these types. Got types: {type(a)} and {type(b)}")

def check_max_diff(a, b, relative=False):
    """Check maximum absolute difference between tensors or pairs of tensors.
    
    Args:
        a: First tensor or iterable of tensors
        b: Second tensor or iterable of tensors to compare against
        relative: Whether to return relative difference

    Returns:
        float or list of floats: Maximum absolute difference(s)
        
    Raises:
        AssertionError: If shapes or devices don't match
        TypeError: If inputs are not both tensors or both iterables
    """
    # Handle single tensors
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        assert a.shape == b.shape, f"Shapes don't match: {a.shape} vs {b.shape}"
        assert a.device == b.device, f"Devices don't match: {a.device} vs {b.device}"
        return torch.abs(a - b).max().item() if not relative else torch.abs(a - b).max().item() / torch.abs(b).max().item()
    elif a is None and b is None:
        return 0.
    elif isinstance(a, (float, int, bool)) and isinstance(b, (float, int, bool)):
        return abs(float(a) - float(b)) if not relative else abs(float(a) - float(b)) / abs(float(b))

    # Handle iterables of tensors
    try:
        return [check_max_diff(a_tensor, b_tensor) for a_tensor, b_tensor in zip(a, b)]
    except TypeError as e:
        if "zip" not in str(e):
            raise
        raise TypeError(f"Inputs must both be tensors or both be iterables of tensors. Got types: {type(a)} and {type(b)}")


def top_k(tensor, k):
    """Return the top k values and indices of a whole tensor"""
    strides = tensor.stride()
    vals, indices = torch.topk(tensor.flatten(), k)
    # reconstruct indices to match original tensor shape
    indices_o = []
    for idx in indices:
        idx = idx.item()
        idx_o = [0] * len(strides)
        for i, stride in enumerate(strides):
            idx_o[i] = idx // stride
            idx = idx % stride
        indices_o.append(tuple(idx_o))
    return vals, indices_o

def report_diff_details(gold, ref, test, tol, atol=0, num_vals=10):
    """Report details about the difference between gold, ref, and test.

    Args:
        gold: Gold tensor or iterable of tensors
        ref: Reference tensor or iterable of tensors
        test: Test tensor or iterable of tensors
        tol: float. Tolerance for difference
        num_vals: int. Number of values to report.

    Returns:
        str: Details about the difference between gold, ref, and test
    """
    if isinstance(gold, torch.Tensor):
        assert isinstance(ref, torch.Tensor) and isinstance(test, torch.Tensor), "If gold is tensor, ref and test must also be tensors"
        gold_ref_diff = torch.abs(gold - ref)
        gold_test_diff = torch.abs(gold - test)
        abs_error = (gold_test_diff - ((1 + tol) * gold_ref_diff + atol))
        violations = (abs_error > 0)
        num_violations = violations.sum().item()
        if num_violations > 0:
            top_abs_errors, abs_error_indices = top_k(abs_error, k=num_vals)
            gold_vals = [gold[*abs_error_indices[i]].item() for i in range(len(abs_error_indices))]
            ref_vals = [ref[*abs_error_indices[i]].item() for i in range(len(abs_error_indices))]
            test_vals = [test[*abs_error_indices[i]].item() for i in range(len(abs_error_indices))]
            rel_error = gold_test_diff / torch.abs(gold)
            violation_pct = num_violations / gold.numel()
            return f"\tViolations: {num_violations}, {violation_pct * 100:.2f}%\nIndices: {abs_error_indices}\nGold: {gold_vals}...\nRef: {ref_vals}...\nTest: {test_vals}... \nMax rel error: {rel_error.max().item():.4f}"
        return ""
    elif hasattr(gold, '__iter__'):
        msgs = [report_diff_details(g, r, t, tol, atol) for g, r, t in zip(gold, ref, test)]
        msg = """\nDiff details:\n"""
        for i, imsg in enumerate(msgs):
            msg += f"\n\tElement {i}/{len(msgs)}:\n\t\t{imsg}"
        return msg
    else:
        raise TypeError(f"Expected tensor or iterable of tensors, got {type(gold)}")
    

def report_diff_details_2(ref, test, tol, atol, num_vals=10):
    """Report details about the difference between ref, and test.

    Args:
        ref: Reference tensor or iterable of tensors
        test: Test tensor or iterable of tensors
        tol: float. Tolerance for difference
        num_val: int. Number of values to report.

    Returns:
        str: Details about the difference between ref, and test
    """
    if isinstance(ref, torch.Tensor):
        assert isinstance(test, torch.Tensor), "If ref is tensor, test must also be tensor"
        ref_test_diff = torch.abs(ref - test)
        violations = (ref_test_diff > torch.abs(ref) * tol + atol)
        if violations.any():
            indices = torch.argwhere(violations)
            ref_vals = [ref[*idx].item() for idx in indices[:num_vals]]
            test_vals = [test[*idx].item() for idx in indices[:num_vals]]
            return f"\tViolations: {len(indices)}\nIndices: {indices}\nRef: {ref_vals}...\nTest: {test_vals}..."
        return ""
    elif hasattr(ref, '__iter__'):
        msgs = [report_diff_details_2(r, t, tol, atol) for r, t in zip(ref, test)]
        msg = """\nDiff details:\n"""
        for i, imsg in enumerate(msgs):
            msg += f"\n\tElement {i}/{len(msgs)}:\n\t\t{imsg}"
        return msg
    else:
        raise TypeError(f"Expected tensor or iterable of tensors, got {type(ref)}")



def report_direct_diff_details(ref, test, tol, num_vals=10):
    """Report details about the difference between ref and test.

    Args:
        ref: Reference tensor or iterable of tensors
        test: Test tensor or iterable of tensors
        tol: float. Tolerance for difference
        num_val: int. Number of values to report.

    Returns:
        str: Details about the difference between gold, ref, and test
    """
    if isinstance(ref, torch.Tensor):
        assert isinstance(test, torch.Tensor), "If ref is tensor, test must also be tensor"
        ref_test_diff = torch.abs(ref - test)
        violations = (ref_test_diff > torch.abs(ref) * tol)
        if violations.any():
            indices = torch.argwhere(violations)
            ref_vals = [ref[*idx].item() for idx in indices[:num_vals]]
            test_vals = [test[*idx].item() for idx in indices[:num_vals]]
            return f"\tViolations: {len(indices)}\nIndices: {indices}\nRef: {ref_vals}...\nTest: {test_vals}..."
        return ""
    elif hasattr(ref, '__iter__'):
        msgs = [report_direct_diff_details(r, t, tol) for r, t in zip(ref, test)]
        msg = """\nDiff details:\n"""
        for i, imsg in enumerate(msgs):
            msg += f"\n\tElement {i}/{len(msgs)}:\n\t\t{imsg}"
        return msg
    else:
        raise TypeError(f"Expected tensor or iterable of tensors, got {type(ref)}")


def check_error_within_tolerance(test_error, *, atol: float = 0., rtol: float = 0.,ref_error: float = None):
    """Check if test error is within tolerance.

    atol is an absolute tolerance threshold, rtol is a relative tolerance threshold as compared to a reference error.
    Rtol does not apply if ref_error is not provided.
    
    Args:
        test_error: Test error level (float) or iterable of test errors to compare
        atol: Absolute tolerance threshold
        rtol: Relative tolerance threshold as compared to reference error, eg .2 means errors up to 20% larger than reference are allowed
        ref_error: Reference error level (float) or iterable of reference errors
        
    Raises:
        AssertionError: If test error exceeds reference by more than tolerance,
            with details about which errors failed and by how much
        TypeError: If inputs are not both floats or both iterables
    """
    # Handle single error case
    if isinstance(test_error, float):
        max_allowed = max((ref_error * rtol) if ref_error is not None else 0., atol)
        if test_error > max_allowed:
            if ref_error is not None:
                raise AssertionError(f"Precision failure: Error ({test_error:.4f}) exceeds reference ({ref_error:.4f}) by more than {max_allowed:.4f} = max({rtol*100:.0f}%, {atol:.0e})")
            else:
                raise AssertionError(f"Precision failure: Error ({test_error:.4f}) exceeds {atol:.0e}")
        return
        
    # Handle iterable case
    try:
        failures = []
        for i, (ref, test) in enumerate(zip(ref_error if ref_error is not None else [None] * len(test_error), test_error)):
            max_allowed = max((ref * rtol) if ref is not None else 0., atol)
            if test > max_allowed:
                if ref is not None:
                    failures.append(f"Element {i}/{len(ref_error)} error ({test:.4f}) exceeds reference ({ref:.4f}) by more than {max_allowed:.4f} = max({rtol*100:.0f}%, {atol:.0e})")
                else:
                    failures.append(f"Element {i}/{len(ref_error)} error ({test:.4f}) exceeds {atol:.0e}")
        if failures:
            raise AssertionError("Precision failure(s): " + ", ".join(failures))
            
    except TypeError:
        raise TypeError("Inputs must both be floats or both be iterables")


def clone_grads(tensor_or_tensors):
    """Clone gradients from tensor(s) that require grad.
    
    Args:
        tensor_or_tensors: A tensor or iterable of items that may be tensors
        
    Returns:
        List of cloned gradients for any tensors that required grad
    """
    # Handle single tensor case
    if isinstance(tensor_or_tensors, torch.Tensor):
        if tensor_or_tensors.requires_grad and tensor_or_tensors.grad is not None:
            return tensor_or_tensors.grad.detach().clone()
        return None
    # Handle iterable case
    grads = []
    for item in tensor_or_tensors:
        if isinstance(item, torch.Tensor) and item.requires_grad:
            if item.grad is not None:
                grads.append(item.grad.detach().clone())
            else:
                grads.append(None)
    return grads


def clone_inputs(things):
    """Clone inputs that are torch tensors or iterables of torch tensors or python objects.

    Args:
        things: A tensor or iterable of tensors or python objects

    Returns:
        List of cloned inputs that are torch tensors or iterables of torch tensors or python objects
    """
    # Handle single tensor case
    if isinstance(things, torch.Tensor):
        t = things.detach().clone()
        if things.requires_grad:
            t.requires_grad = True
            t.retain_grad()
        return t
    elif not hasattr(things, '__iter__'):
        return deepcopy(things)
    elif hasattr(things, '__iter__'):
        return [clone_inputs(item) for item in things]
    else:
        raise TypeError(f"Expected tensor or iterable of tensors, got {type(things)}")


def create_random_grads(tensor_or_tensors, gold_grads=None, scale=1.0, seed=42):
    """Create random gradients for tensor(s), if gold_grads is not provided.
       Create copy of gold_grads that has same shape, dtype, and device as input.
    
    Args:
        tensor_or_tensors: A tensor or iterable of tensors
        grads: Optional list of gradients to copy
        scale: Scale factor for random gradient
    Returns:
        List of random gradients with matching shape, dtype, and device as input
    """
    torch.manual_seed(seed)
    if isinstance(tensor_or_tensors, torch.Tensor):
        if gold_grads is None:
            return torch.randn(tensor_or_tensors.shape, dtype=tensor_or_tensors.dtype, device=tensor_or_tensors.device) * scale
        else:
            assert isinstance(gold_grads, torch.Tensor), "gold_grads must be a tensor"
            return gold_grads.detach().clone().to(tensor_or_tensors.device).to(tensor_or_tensors.dtype) * scale
    elif hasattr(tensor_or_tensors, '__iter__') and not isinstance(tensor_or_tensors, torch.Tensor):
        if gold_grads is None:
            grads = []
            for item in tensor_or_tensors:
                if isinstance(item, torch.Tensor):
                    grads.append(torch.randn(item.shape, dtype=item.dtype, device=item.device) * scale)
                else:
                    grads.append(None)
        else:
            assert hasattr(gold_grads, '__iter__'), "gold_grads must be an iterable"
            grads = []
            for item, gold_grad in zip(tensor_or_tensors, gold_grads):
                if isinstance(item, torch.Tensor):
                    tmp = torch.empty_like(item)
                    tmp[:] = gold_grad.detach().clone().to(item.device).to(item.dtype)[:] * scale
                    grads.append(tmp)
                else:
                    grads.append(None)
        return grads
    else:
        raise TypeError(f"Expected tensor or iterable of tensors, got {type(tensor_or_tensors)}")


def clear_grads(tensor_or_tensors):
    """Clear gradients from tensor(s) that require grad.
    
    Args:
        tensor_or_tensors: A tensor or iterable of items that may be tensors
    """
    # Handle single tensor case
    if isinstance(tensor_or_tensors, torch.Tensor):
        if tensor_or_tensors.requires_grad:
            tensor_or_tensors.grad = None
        return
    # Handle iterable/dict case
    if isinstance(tensor_or_tensors, dict):
        for item in tensor_or_tensors.values():
            if isinstance(item, torch.Tensor) and item.requires_grad:
                item.grad = None
    else:
        for item in tensor_or_tensors:
            if isinstance(item, torch.Tensor) and item.requires_grad:
                item.grad = None


def sanity_check(tensors):
    """ Perform a sanity check on a tensor or list of tensors.
    """
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            if tensor.isnan().any():
                nan_indices = torch.argwhere(tensor.isnan())
                raise ValueError(f"Tensor {i}/{len(tensors)} contains NaN values at {len(nan_indices)}/{tensor.numel()} indices {nan_indices}")
            if tensor.isinf().any():
                inf_indices = torch.argwhere(tensor.isinf())
                raise ValueError(f"Tensor {i}/{len(tensors)} contains Inf values at {len(inf_indices)}/{tensor.numel()} indices {inf_indices}")
            if (tensor.abs() > 1e12).any():
                large_indices = torch.argwhere(tensor.abs() > 1e12)
                raise ValueError(f"Tensor {i}/{len(tensors)} contains values greater than 1e12 at {len(large_indices)}/{tensor.numel()} indices {large_indices} {tensor[large_indices]}")


def assert_close_and_report(ref, test, rtol=1e-3, atol=1e-5, pct_threshold=0, num_threshold=0):
    if isinstance(ref, torch.Tensor):
        diff = (ref - test).abs()
        threshold = ref.abs() * rtol + atol
        error_pct = (diff > threshold).float().mean().item()
        num_errors = (diff > threshold).sum().item()
        if error_pct > pct_threshold and num_errors > num_threshold:
            msg = f"Error percentage: {error_pct*100:.3f}%" + report_diff_details_2(ref, test, rtol, atol)
            raise AssertionError(msg)
    elif hasattr(ref, '__iter__'):
        msgs = []
        for i, (r, t) in enumerate(zip(ref, test)):
            diff = (r - t).abs()
            threshold = r.abs() * rtol + atol
            error_pct = (diff > threshold).float().mean().item()
            error_num = (diff > threshold).sum().item()
            if error_pct > pct_threshold and error_num > num_threshold:
                msgs.append(f"Element {i}/{len(ref)}: error percentage: {error_pct*100:.3f}%" + report_diff_details_2(r, t, rtol, atol))
        if msgs:
            raise AssertionError("\n".join(msgs))
    else:
        raise TypeError(f"Expected tensor or iterable of tensors, got {type(ref)}")


def check_inputs_created_determinstically(create_inputs, kwargs, debug_fn=None):
    first_inputs = create_inputs(**kwargs)
    second_inputs = create_inputs(**kwargs)
    if debug_fn is not None:
        debug_fn(first_inputs, second_inputs)
    check_all_equivalent(first_inputs, second_inputs)

def check_fn_compiles(fn, inputs, rtol=None, atol=None):
    with torch.no_grad():
        output_or_outputs = fn(*inputs.values())    
    compiled_fn = torch.compile(fn)
    with torch.no_grad():
        for _ in range(3):
            output_or_outputs_from_compiled = compiled_fn(*inputs.values()) # TODO(jbuckman): horrible hack to get torch.compile to work
    if rtol is not None or atol is not None:
        check_all_equivalent(output_or_outputs, output_or_outputs_from_compiled, rtol=rtol, atol=atol)
    torch._dynamo.reset()

def check_fn_compiles_with_backward(fn, inputs, rtol=1e-3, atol=1e-3, ignore_nan=False, seed=42):
    def fwd_bwd():
        output_or_outputs = fn(**inputs)
        output_grads = torch.ones_like(output_or_outputs) if isinstance(output_or_outputs, torch.Tensor) else [torch.ones_like(o) for o in output_or_outputs]
        torch.autograd.backward(output_or_outputs, grad_tensors=output_grads, retain_graph=True)
        return clone_grads(inputs)
    grads = fwd_bwd()
    compiled_fn = torch.compile(fwd_bwd)
    for _ in range(2):
        clear_grads(inputs)
        grads_from_compiled = compiled_fn()
    check_tensors_properties(grads, grads_from_compiled)
    if rtol is not None or atol is not None:
        check_all_equivalent(grads, grads_from_compiled, rtol=rtol, atol=atol, ignore_nan=ignore_nan)
    torch._dynamo.reset()

def check_fake_fn_implementation_matches(fn, fake_fn, inputs):
    real_output = fn(**inputs)
    fake_output = fake_fn(**inputs)
    check_tensors_properties(real_output, fake_output)


def check_inputs_forwards_match(*, fn, inputs1, inputs2, atol=1e-5, verbose=False):
    """Given a function, check that it produces the same output for two sets of inputs.
    
    Args:
        fn: Function to check
        inputs1: First set of inputs for function
        inputs2: Second set of inputs for function
        atol: float. Absolute tolerance for difference
    """
    with torch.no_grad():
        output1 = fn(**inputs1)
        output2 = fn(**inputs2)
    sanity_check(output1)
    sanity_check(output2)

    if verbose:
        def find_first_mismatch(a, b, atol):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                diff = torch.abs(a - b)
                if (diff > atol).any():
                    idx = (diff > atol).nonzero()[0]
                    print(f"First mismatch at index {tuple(idx.tolist())}: {a[tuple(idx)].item():.6f} vs {b[tuple(idx)].item():.6f}")
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                for a_item, b_item in zip(a, b):
                    find_first_mismatch(a_item, b_item, atol)
        find_first_mismatch(output1, output2, atol)

    err = check_max_diff(output1, output2)
    check_error_within_tolerance(err, atol=atol)

def check_inputs_backwards_match(*, fn, inputs1, inputs2, atol=1e-5):
    """Given a function, check that it produces the same gradients for two sets of inputs.
    
    Args:
        fn: Function to check
        inputs1: First set of inputs for function
        inputs2: Second set of inputs for function
        atol: float. Absolute tolerance for difference
    """
    output1 = fn(**inputs1)
    torch.autograd.backward(output1, grad_tensors=torch.ones_like(output1))
    grads1 = clone_grads(inputs1)

    output2 = fn(**inputs2)
    torch.autograd.backward(output2, grad_tensors=torch.ones_like(output2))
    grads2 = clone_grads(inputs2)

    sanity_check(grads1)
    sanity_check(grads2)
    err = check_max_diff(grads1, grads2)
    check_error_within_tolerance(err, atol=atol)

def check_fn_forwards_match(*, ref_fn, gold_inputs, test_fn, test_inputs, rtol=.1, atol=1e-5):
    """Given two functions, check that they produce the same output for the same inputs.

    gold_inputs are high-precision inputs, the reference function is run with these inputs
    to produce a gold output. Then both the reference and test functions are run with
    the low-precision test_inputs to produce a reference and test output.
    
    Args:
        ref_fn: Reference function
        gold_inputs: High-precision inputs for reference function
        test_fn: Test function
        test_inputs: Low-precision inputs for reference & test function
        tol: float. Tolerance for difference
        atol: float. Absolute tolerance for difference
    """
    # ref_inputs = clone_inputs(test_inputs)
    ref_inputs = test_inputs
    with torch.no_grad():
        gold_output = ref_fn(**gold_inputs)
        ref_output = ref_fn(**ref_inputs)
        test_output = test_fn(**test_inputs)
    sanity_check(gold_output)
    sanity_check(ref_output)
    sanity_check(test_output)
    ref_err = check_max_diff(gold_output, ref_output)
    test_err = check_max_diff(gold_output, test_output)
    check_error_within_tolerance(test_err, atol=atol, rtol=rtol, ref_error=ref_err)

def check_fn_backwards_match(*, ref_fn, gold_inputs, test_fn, test_inputs, rtol=.1, atol=1e-5):
    """Given two functions, check that they produce the same gradients for the same inputs.

    gold_inputs are high-precision inputs, the reference function is run with these inputs
    to produce a gold output. Then both the reference and test functions are run with
    the low-precision test_inputs to produce a reference and test output.
    
    Args:
        ref_fn: Reference function
        gold_inputs: High-precision inputs for reference function
        test_fn: Test function
        test_inputs: Low-precision inputs for reference & test function
        tol: float. Tolerance for difference
        atol: float. Absolute tolerance for difference
    """

    gold_output = ref_fn(**gold_inputs)
    torch.autograd.backward(gold_output, grad_tensors=torch.ones_like(gold_output))
    gold_grads = clone_grads(gold_inputs)

    ref_output = ref_fn(**test_inputs)
    torch.autograd.backward(ref_output, grad_tensors=torch.ones_like(ref_output))
    ref_grads = clone_grads(test_inputs)
    
    clear_grads(test_inputs)
    test_output = test_fn(**test_inputs)
    torch.autograd.backward(test_output, grad_tensors=torch.ones_like(test_output))
    test_grads = clone_grads(test_inputs)

    sanity_check(gold_grads)
    sanity_check(ref_grads)
    sanity_check(test_grads)
    ref_err = check_max_diff(gold_grads, ref_grads)
    test_err = check_max_diff(gold_grads, test_grads)
    check_error_within_tolerance(test_err, atol=atol, rtol=rtol, ref_error=ref_err)
    torch.cuda.empty_cache()
