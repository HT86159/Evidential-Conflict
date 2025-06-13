import argparse
import torch
import sys
sys.path.append("../models/LLaVA/")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from tqdm import tqdm
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import pandas as pd
import math
import os
from scipy.spatial.distance import cosine



def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


save_name="-llava-10000_with_baselines-4"

save_device = "cuda:1"




def load_mydataset():
    path='/data/public/data/IJAR-dataset'
    df = pd.read_excel(path+'/dataset.xlsx', sheet_name='Filtered_Data')
    data_list = df.to_dict(orient='records')
    images=[]
    for item in data_list:
        images.append(path+'/images'+str(item['image']))
    return list(zip(data_list,images))



def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit,args.load_4bit, device=args.device)


    datasets=load_mydataset()[4500:10000]



    flag=0
    total_index=[]


    for batch in tqdm(datasets):

        flag+=1
        options={'A':batch[0]['A'],'B':batch[0]['B']}
        # if not math.isnan(batch[0]['C']):
        #     options['C']=batch[0]['C']
        # if not math.isnan(batch[0]['D']):
        #     options['D']=batch[0]['D']
        if isinstance(batch[0]['C'], (int, float)) and not math.isnan(batch[0]['C']):
            options['C'] = batch[0]['C']
        elif isinstance(batch[0]['C'], str) and batch[0]['C'] != "":
            options['C'] = batch[0]['C']

        # 检查 'D' 是否不是NaN（仅限数字类型）且不是空字符串
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

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image = load_image(image_file)
        image_size = image.size
        # Similar operation in model_worker.py

        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)





        inp=DEFAULT_IMAGE_TOKEN+query
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        full_responses=[]

        for iter in range(6):
            if iter==0:
                temperature=0.7
            else:
                temperature=1.0

            with torch.inference_mode():
    
                output_ids,all_probs,probs_dis,next_token_evidential_activation,down_proj_input_list  = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    output_hidden_states=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = outputs
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



        if flag % 10 == 0:
            print('当前轮数为:',flag)
            torch.save(total_index, os.path.join("../infer_results", save_name+".pth"))
            print("save {}-{}.pth".format(save_name, flag))
            torch.cuda.empty_cache() # 清空缓存



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default=save_device)
    parser.add_argument("--conv-mode", type=str, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", default= True,action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
