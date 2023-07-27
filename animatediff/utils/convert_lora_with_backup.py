from safetensors.torch import load_file
from collections import defaultdict
import torch 
import os

loaded_loras = None
original_weight = {}
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
network_updates = defaultdict(dict)

def load_loras(pipeline, loras, device):
    global loaded_loras, network_updates
    if loras == loaded_loras:
        print("No Loras changed")
    else:
        if loaded_loras is not None:
            restore_networks(pipeline, network_updates)

        # Reset network updates for new Lora weights
        network_updates = defaultdict(dict)
        for lora in loras:
            lora_path = lora['path']
            # Check if path exists
            if lora_path == "none" or not os.path.exists(lora_path):
                continue
            state_dict = load_file(lora_path, device=device)
            # Getting all network updates
            network_updates = get_network_updates(network_updates, state_dict=state_dict)
        backup_networks(pipeline, network_updates)
        for lora in loras:
            lora_path = lora['path']
            # Check if path exists
            if lora_path == "none" or not os.path.exists(lora_path):
                continue
            lora_alpha = lora['alpha']
            print(f"loading lora {lora_path} with weight {lora_alpha}")
            state_dict = load_file(lora_path, device=device)
            backup_networks(pipeline, state_dict)
            pipeline = load_lora_weights(pipeline, state_dict, lora_alpha, device, torch.float32)
            loaded_loras = loras
    return pipeline


def get_network_updates(network_updates, state_dict):
    
    for key, value in state_dict.items():
    # it is suggested to print out the key, it usually will be something like below
    # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        network_updates[layer][elem] = value

    return network_updates

def get_target_layer(layer, pipeline):
    if "text" in layer:
        layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
        curr_layer = pipeline.text_encoder
    else:
        layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
        curr_layer = pipeline.unet

    # find the target layer
    temp_name = layer_infos.pop(0)
    while len(layer_infos) > -1:
        try:
            print(temp_name)
            curr_layer = curr_layer.__getattr__(temp_name)
            if len(layer_infos) > 0:
                temp_name = layer_infos.pop(0)
            elif len(layer_infos) == 0:
                break
        except Exception:
            if len(temp_name) > 0:
                temp_name += "_" + layer_infos.pop(0)
            else:
                temp_name = layer_infos.pop(0)
    return curr_layer


def backup_networks(pipeline, network_updates):
    global original_weight
    for layer, elems in network_updates.items():
        curr_layer = get_target_layer(layer, pipeline)
        original_weight[layer] = curr_layer.weight.data.clone().detach()
    return pipeline

def restore_networks(pipeline, network_updates):
    global original_weight
    for layer, elems in network_updates.items():
        curr_layer = get_target_layer(layer, pipeline)
        # if layer in original_text_encoder:
        curr_layer.weight.data = original_weight[layer].clone().detach()

    return pipeline


# Modified code from https://github.com/huggingface/diffusers/issues/3064#issuecomment-1514082155
def load_lora_weights(pipeline, state_dict, multiplier, device, dtype):
    # if (pipeline != current_pipeline):
    #     backup = True
    #     print("Backing up weights")
    #     current_pipeline = pipeline
    #     original_weights = {}    
    # else:
    #     backup = False
    #     print("Not Backing up weights")
    
    # load base model
    pipeline.to(device)

    # load LoRA weight from .safetensors
    network_updates = defaultdict(dict)
    network_updates = get_network_updates(network_updates=network_updates, state_dict=state_dict)

    for layer, elems in network_updates.items():
        curr_layer = get_target_layer(layer, pipeline)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0
        
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

# LoraLoaderMixin.load_lora_weights = load_lora_weights