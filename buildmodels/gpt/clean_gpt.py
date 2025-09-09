import onnx

from replace_gelu import replace_gelu
from replace_rope import replace_for_rope, replace_for_rope_mul
from replace_multihead import replace_for_multihead

model_path = 'buildmodels/gpt/nano_gpt_2_layers_8_embd.onnx'

model = onnx.load(model_path, load_external_data=False)

model = replace_gelu(model)
model = replace_for_rope(model)
model = replace_for_rope_mul(model)

model = replace_for_multihead(model)
onnx.save(model, model_path)