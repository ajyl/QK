import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

key_module_name = "model.layers.{}.self_attn.hook_key_states"
value_module_name = "model.layers.{}.self_attn.hook_value_states"
query_module_name = "model.layers.{}.self_attn.hook_query_states"
attn_module_name = "model.layers.{}.self_attn.hook_attn_pattern"
qk_module_name = "model.layers.{}.self_attn.hook_qk_logits"
resid_mid_module_name = "model.layers.{}.hook_resid_mid"
resid_post_module_name = "model.layers.{}.hook_resid_post"
