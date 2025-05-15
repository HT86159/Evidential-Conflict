
from utils import get_files_with_prefix
from config.vicuna.fastchat.model.model_adapter import load_model
import numpy as np
import torch
from transformers import GPTJForCausalLM, AutoTokenizer,AutoModelForSequenceClassification, AutoTokenizer
from collections import defaultdict, Counter
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
import torch.nn.functional as F
from llama_tokenizer import Tokenizer
from torch.nn.functional import cosine_similarity
import spacy
import re
import tqdm
from rouge_score import rouge_scorer
import os
import pandas as pd
import math
from beliefNN import EvidenceModel
# from evidencemodel1 import EvidenceModel 
# import NNbelief_torch as nnbelief
import math
from mpmath import mp, log
from collections import Counter
from typing import List, Tuple, Optional



from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama


device = "cuda:1"
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:1")

llh_shift = 1e-6
entailment_model = EntailmentDeberta()


    

def sum_probs(probs_list):
    return [torch.sum(sum(probs_list))]

def multiply_probs(probs_list):
    return [torch.sum(torch.log(probs.float()+llh_shift)).item() for probs in probs_list]

def predeictive_entropy(original_prob_distributions):
    """
    Calculate predictive entropy (PE) of a sample.
    """
    out = []
    # import   pdb; pdb.set_trace()
    for prob_dis in original_prob_distributions:
        # sentence_ent = sum([-(x * torch.log(x.float() + llh_shift)).sum() for x in prob_dis])
        sentence_ent = sum([-(torch.log(x.float() + llh_shift)).sum() for x in prob_dis])
        # sentence_ent=-1*sum([x*torch.log(x.float()+llh_shift) for x in prob_dis])
        # import   pdb; pdb.set_trace()
        out.append(sentence_ent.item())
  
    return out


def ln_predictive_entropy(original_prob_distributions,lengths):
    # import pdb;pdb.set_trace()
    """
    Calculate Length-normalised predictive entropy (LN_PE) of a sample.
    """
    return torch.tensor(predeictive_entropy(original_prob_distributions))/torch.tensor(lengths)

def negative_log(probs):
    # import pdb;pdb.set_trace()
    negative_log_list=[-1*torch.log(p.float()) for p in probs]
    return max(negative_log_list[0]).item()


def semantic_entropy(question,full_responses,example):
    responses = [fr[0] for fr in full_responses]
    log_liks=[r[1] for r in full_responses]
    # responses = [f'{question} {r}' for r in responses] 
    # import pdb;pdb.set_trace()
    semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=True, example=example)

    log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized') #每个semantic id的log p
    pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
    return pe

def compute_self_consistency(outputs: List[str]) -> Optional[Tuple[str, int]]:
    if not outputs:
        return None

    counter = Counter(outputs)
    most_common_output, frequency = counter.most_common(1)[0]
    return most_common_output, frequency

def evidence_metrics(model_weights,evidence_activations):
    model_weights=model_weights.to(device)
    model_weights=model_weights.T
    bias = torch.zeros(model_weights.size(1)).to(device)
    evidence_model = EvidenceModel(model_weights,bias)
    evidence_weights = evidence_model.get_evidence_weights(evidence_activations.squeeze(0).T)
    conflict_value = evidence_model.get_evidence_conflict()
    return conflict_value


prefix = "pope-adversarial-llava"

files_list = get_files_with_prefix(prefix)
print(files_list)
# model_weights=torch.load('../model_weights/llava_lm_head_weights.pth').to(device)
# model_weights=torch.load('../model_weights/mplug_owl2_lm_head_weights.pth').to(device)
model_weights=torch.load('../model_weights/mplug_owl3_lm_head_weights.pth').to(device)
bias=torch.zeros(model_weights.size(0)).to(device)
# files_list = ["/data/liuzhekun1/projects/IJAR/infer_results/-llava-10000_with_baselines.pth"]
# files_list = ["/data/liuzhekun1/projects/IJAR/infer_results/-mplug_owl2-10000_with_baselines.pth"]
files_list = ["/data/liuzhekun1/projects/IJAR/infer_results/-mplug_owl3-10000_with_baselines.pth"]
output_dir = "/data/liuzhekun/liuzhekun/projects/IJAR/infer_results"




for file in tqdm.tqdm(files_list):
    print('flie为:',file)
    file_name = file.split("/")[-1].split(".")[0]
    output_file = os.path.join(output_dir,file_name.replace(".pth", "_uncertainty.pth"))
    if os.path.exists(output_file):
        continue
    print(f"processing {file}")

    uncertainty_list = []
    data = torch.load(file)






    for i,(input) in tqdm.tqdm(enumerate(data[0:5000])):


        uncertainty_dict = dict()

        labels=[]
        sample = input["original_data"]
        original_questions = sample["original_questions"]
        gt_answers = sample["gt_answers"]
        original_answers = sample["original_answers"]
        original_prob_distributions = sample["original_prob_distributions"]
   
        # token_importance_scores=sample['tokens_importance_scores']

        lengths = sample["lengths"]
        evidence_activations = sample["evidence_activations"]
        full_responses=input['reference_data']


        


        if "vicuna" in prefix:
            original_probs = sample["original_probs"]
            original_probs = [item for item in original_probs if not (isinstance(item, list) and len(item) == 0)]
        else:
            original_probs = [torch.stack(x) if len(x) > 0 else x for x in sample["original_probs"]]


        example=dict()
        example['question']=original_questions[0]
        example['context']=''
        example['most_likely_answer']={'response':original_answers[0],'token_log_likelihoods':original_probs[0].tolist(),'probs_distributions':original_prob_distributions[0],'length':lengths[0],'embedding':None}

        

        
        

        conflict_list=[]

        # import pdb;pdb.set_trace()
        for evidence in evidence_activations[0]:
            evidence=evidence.to(device)
            conflict_value=evidence_metrics(model_weights,evidence)
            conflict_list.append(conflict_value.detach().cpu().numpy().tolist())


        outputs=[]
        outputs.append(example['most_likely_answer']['response'])
        for item in full_responses:
            outputs.append(item[0])


        probs_sum = sum_probs(original_probs)
        probs_multiply=multiply_probs(original_probs)
        pe = predeictive_entropy(original_prob_distributions)
        ln_pe = ln_predictive_entropy(original_prob_distributions,lengths)
        negative_log_value=negative_log(original_probs)

        # if i==130:
        #     import pdb;pdb.set_trace()
        # else:
        #     continue
        semantic_entropy_value=semantic_entropy(original_questions[0],full_responses,example)

        sc=1-(compute_self_consistency(outputs)[1]/len(outputs))



        conflict_max=max([float(c[0]) for c in conflict_list])
        conflict_sum=sum([float(c[0]) for c in conflict_list])
        conflict_avg=conflict_sum/ len(conflict_list)

              
        uncertainty_dict["probs_sum"] = torch.tensor(probs_sum)
        uncertainty_dict['probs_multiply']=torch.tensor(probs_multiply)
        uncertainty_dict["pe"] = torch.tensor(pe)
        uncertainty_dict["ln_pe"] = ln_pe
        uncertainty_dict['negative_log']=negative_log_value
        uncertainty_dict['conflict_max']=conflict_max
        uncertainty_dict['conflict_sum']=conflict_sum
        uncertainty_dict['conflict_avg']=conflict_avg
        uncertainty_dict['semantic_entropy']=semantic_entropy_value
        uncertainty_dict['self_consistency']=sc
        uncertainty_dict["labels"] = labels
        uncertainty_dict["original_questions"] = original_questions
 
        uncertainty_list.append(uncertainty_dict)
        del probs_sum, probs_multiply, pe,ln_pe, negative_log_value, conflict_max,conflict_sum,conflict_avg,semantic_entropy_value,original_questions,sc

    torch.save(uncertainty_list, file.replace(".pth","_uncertainty_1.pth"))