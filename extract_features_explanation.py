from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import pandas as pd
import torch
import json
import os
import re
import requests
import argparse
from sae_lens import SAE  
from transformer_lens import HookedTransformer

# helper function to filter the target explanations
def filter_explanation(feature_id, explanations, desc_only = False):
  """
  Helper function to filter the desired feature explanations
  feature_id: int : id of the target sae feature
  explanations: List[Dict]: the set of downloaded explanations to use
  desc_only: bool: return all metadata or only the textual description
   """
  metadata = [e for e in explanations if int(e["index"]) == feature_id][0]
  if not desc_only:
    return metadata
  else:
    return metadata["description"].strip()
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# torch.cuda.set_manual_seed(42)
print(DEVICE)




if __name__ == "__main__()":
    # parser = argparse.ArgumentParser()
    # #TODO add arguments
    # parser.add_argument("-m", "--model", help= "model to analyse")
    # parser.add_argument("-b", "--block", help = "the transfomer block (layer) to consider ")
    # parser.add_argument("-l", "--location",
    #                      help = "the location inside the transformer block to extract activation from [attn_output, mlp_out, resid_post]")
    # parser.parse_args()
    
    target_location =  'blocks.25.hook_resid_post' #parser.location
    
    #load the model with translens
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device="cuda")
    #load sae  from sae lens
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = "layer_25/width_16k/canonical")
    
    # load data
    data  = json.load(open("animals.json", "r"))

    #format text
    template = "The {hypo} is a {hyper}"
    input_texts = [template.format(a) for a in data if model.to_single_token(a)]
    # extract all activation from the given location for each input
    target_act_all = [model.run_with_cache(i)[1][target_location].squeeze() for i in input_texts]
    # gather sae activations
    sae_act_all = [sae.encode(t) for t in target_act_all]
    #get indices of sparse activated features
    indices = np.stack([act.max(-1).indices.detach().cpu().numpy() for act in sae_act_all])
    # store the topk actiating features
    top_k_ids = [act[2].topk(5).indices.detach().cpu().numpy() for act in sae_act_all]

    gs_explanations = requests.get("https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-2b_25-gemmascope-res-16k.json").json()
    
    #extract explanations for all top k features
    explanations = []
    for id_set in top_k_ids:
        expl = []
        for id in id_set: #range(id_set), id_set[id]
            expl.append(filter_explanation(id, gs_explanations, desc_only = True))
        explanations.append(expl)
        # explanations.append(expl.insert(0, input_texts[id]))
    print(len(explanations))

    df = pd.DataFrame(explanations, columns = [f"Top{k}_feature" for k in range(1,6)], index = [",".join((a[0],i)) for a,i in zip(pairs,input_texts)])
    df.to_csv("./topk_explanations2.tsv", sep = "\t")
    
    print("Saved a dataframe containing explanations")