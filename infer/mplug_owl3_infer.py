import torch
from modelscope import AutoConfig, AutoModel


save_name="-mplug_owl3-10000_with_baselines-4"
device = "cuda:7"

model_path = 'iic/mPLUG-Owl3-7B-241101'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(config)
model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True)
_ = model.eval().cuda(device)





from PIL import Image
import pandas as pd
from modelscope import AutoTokenizer
from decord import VideoReader, cpu 
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)
from tqdm import tqdm
import math
import os

def load_mydataset():
    path='/data/public/data/IJAR-dataset'
    df = pd.read_excel(path+'/dataset.xlsx', sheet_name='Filtered_Data')
    data_list = df.to_dict(orient='records')
    images=[]
    for item in data_list:
        images.append(path+'/images'+str(item['image']))
    return list(zip(data_list,images))


datasets=load_mydataset()[4500:10000]


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


    image = Image.open(image_file).convert('RGB')

    messages = [
        {"role": "user", "content": f"<|image|>{query}"},
        {"role": "assistant", "content": ""}
    ]

    inputs = processor(messages, images=[image], videos=None)

    inputs.to(device)
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':100,
        'decode_text':True,
    })

    full_responses=[]
    for iter in range(6):
        with torch.inference_mode():
            outputs,all_probs,probs_dis,next_token_evidential_activation,down_proj_input_list = model.generate(**inputs)
        original_answer=outputs[0]
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

        original_prob=[x.to(device) for x in original_prob]
        original_prob_distribution=[x.to(device) for x in original_prob_distribution]
        evidence_activation = [x.to(device) for x in evidence_activation]
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
