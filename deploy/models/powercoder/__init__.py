from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from .configuration_powercoder import PowerCoderConfig
from .modeling_powercoder import PowerCoderForCausalLM

# make HF aware of the new model
AutoConfig.register("powercoder", PowerCoderConfig)
AutoModelForCausalLM.register(PowerCoderConfig, PowerCoderForCausalLM)

AutoTokenizer.register(PowerCoderConfig, GPT2Tokenizer, GPT2TokenizerFast)

__all__ = ["PowerCoderConfig", "PowerCoderForCausalLM"]