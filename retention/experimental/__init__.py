from retention.vidrial import power_retention, power_retention_inference
from retention.create_inputs import create_inputs, create_inference_inputs, inference_input_properties, inference_output_properties, input_properties, output_properties
from vidrial.kernels.flash.interface import flash
import logging

logging.warning("Using experimental features of retention. Experimental features may not be bug-free and may be changed without notice. Use at your own discretion.")

__all__ = [
    'power_retention',
    'power_retention_inference',
    'create_inputs',
    'input_properties',
    'output_properties',
    'create_inference_inputs',
    'inference_input_properties',
    'inference_output_properties',
    'flash',
]
