import sys
sys.path.append("../models/mPLUG-Owl/mPLUG-Owl2/")
print(sys.path)
from tqdm import tqdm
import torch
from PIL import Image
from transformers import TextStreamer
import time
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import pandas as pd
import math
import os
import json


save_name="-mplug_owl2-10000_with_baselines_new_2-5"

save_device = "cuda:7"


def load_mydataset():
    path='/data/public/data/IJAR-dataset'
    df = pd.read_excel(path+'/dataset.xlsx', sheet_name='Filtered_Data')
    data_list = df.to_dict(orient='records')
    images=[]
    for item in data_list:
        images.append(path+'/images'+str(item['image']))
    return list(zip(data_list,images))




model_path = '/data/liuzhekun1/projects/DS-multimodal-hallucination/VLM-test/models/mPLUG-Owl1'


model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_4bit=True, device=save_device)




datasets=load_mydataset()[2571:2857]



flag=0
total_index=[]
for batch in tqdm(datasets):
    flag+=1
    options={'A':batch[0]['A'],'B':batch[0]['B']}
    if isinstance(batch[0]['C'], (int, float)) and not math.isnan(batch[0]['C']):
        options['C'] = batch[0]['C']
    elif isinstance(batch[0]['C'], str) and batch[0]['C'] != "":
        options['C'] = batch[0]['C']
    if isinstance(batch[0]['D'], (int, float)) and not math.isnan(batch[0]['D']):
        options['D'] = batch[0]['D']
    elif isinstance(batch[0]['D'], str) and batch[0]['D'] != "":
        options['D'] = batch[0]['D']

    # query = batch[0]['question']  + str(options)+'please answer the question with the option letter (A, B, C or D) as detailed as possible.'
    query = batch[0]['question']  + str(options)+'please answer the question as detailed as possible.'
 
    image_file=batch[1]
    
    original_answers=[]
    original_probs=[]
    original_prob_distributions = []
    original_questions = []
    lengths = []
    evidence_activations = []
    gt_answers = []



    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles



    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    
    max_new_tokens = 512


    full_responses=[]

    for iter in range(6):
        if iter==0:
            temperature=0.7
        else:
            temperature=1.0

        with torch.inference_mode():
            output_ids,all_probs,probs_dis,next_token_evidential_activation,down_proj_input_list = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_hidden_states=True)
            
        import pdb;pdb.set_trace()
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith('</s>'):
                        outputs = outputs[:-len('</s>')]
        original_answer=outputs.strip()

        token_log_likelihoods = [torch.log(p).item() for p in all_probs]
        embedding=None

        acc=None

        original_prob_distribution=probs_dis
        original_prob=[x.max() for x in original_prob_distribution]
        # import pdb;pdb.set_trace()
        evidence_activation=next_token_evidential_activation
        evidence_activation=[x[:,-1:,:].cpu() for x in evidence_activation]
        # mlp=down_proj_input_list
        gt_answer=batch[0]['answer']
        hallucination_type=batch[0]['hallucination_type']
        length=len(original_prob)

        original_prob=[x.to(save_device) for x in original_prob]
        original_prob_distribution=[x.to(save_device) for x in original_prob_distribution]
        evidence_activation = [x.to(save_device) for x in evidence_activation]
        # mlp=[x.to(save_device) for x in mlp]

        if iter==0:
            original_questions.append(query)
            lengths.append(length)
            original_prob_distributions.append(original_prob_distribution)
            evidence_activations.append(evidence_activation)
            # mlps.append(mlp)
            gt_answers.append(gt_answer)
            original_answers.append(original_answer)
            original_probs.append(original_prob)

            del original_prob, original_prob_distribution, evidence_activation
            torch.cuda.empty_cache()




            sample = dict()
            sample["gt_answers"] = gt_answers
            sample['hallucination_type']=hallucination_type
            sample["original_answers"] = [s.replace('</s>', '') for s in original_answers]
            sample["original_probs"] = original_probs
            sample["original_prob_distributions"] = original_prob_distributions
            sample["original_questions"] = original_questions
            sample["lengths"] = lengths
            sample["evidence_activations"] = evidence_activations
            out=dict()
            out["original_data"] = sample
            out["reference_data"] = []
            del sample
        
        else:
            out["reference_data"].append((original_answer, token_log_likelihoods, embedding, acc))









    for i in range(len(out['original_data']['original_probs'][0])):
        out['original_data']['original_probs'][0][i]=out['original_data']['original_probs'][0][i].to('cpu')
    for i in range(len(out['original_data']['original_prob_distributions'][0])):
        out['original_data']['original_prob_distributions'][0][i]=out['original_data']['original_prob_distributions'][0][i].to('cpu')
    for i in range(len(out['original_data']['evidence_activations'][0])):
        out['original_data']['evidence_activations'][0][i]=out['original_data']['evidence_activations'][0][i].to('cpu')
    # for i in range(len(out['original_data']['mlps'][0])):
    #     out['original_data']['mlps'][0][i]=out['original_data']['mlps'][0][i].to('cpu')


    total_index.append(out)
    del out

    # if flag % 10 == 0:
    if flag % 10 == 0 or flag==1:
        print('当前轮数为:',flag)
        # import pdb;pdb.set_trace()
        torch.save(total_index, os.path.join("../infer_results", save_name+".pth"))
        print("save",os.path.join("../infer_results", save_name+".pth"))
        torch.cuda.empty_cache() # 清空缓存




  
    

