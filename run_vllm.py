import os, json, glob, logging, sys
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from lacebench.metric import compute_metrics_custom, compute_acc
from lacebench.utils import *
from lacebench import ROOT_PATH, CAPTION_PATH

logger = logging.getLogger(__name__)


# ---------------------- CONFIG ----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")  # 환경변수로 지정 가능
PROMPT = "Describe the given image in one sentence."
CAP_PROMPT = "Describe a sentence for the given image."
PAP_PROMPT = "The given image defines several objects as a group and creates a bounding box. Describe this bounding box at paragraph level. Paragraphs must consist of at least three sentences."
EDIT_PROMPT = "In the given [caption], make sure to generate only the words without article that correspond to the {object}"

system_message = """You are a Vision Language Model specialized in image captioning.
Your task is to analyze the provided image and generate an appropriate caption.
The image is cleared only in the region corresponding to the caption, and all other parts are blurred.
Focus on delivering accurate, succinct captions based on the visual information. Avoid additional explanation unless absolutely necessary."""

LOAD_ADAPTER = False
ADAPTER_PATH = f"./outputs/{MODEL_NAME}_LRGR_bbox/checkpoint-candi-1/"
USE_VLLM = False 
BATCH_SIZE = 1
KCC=False

INCLUDE_BBOX=True
KNOWLEDGE_EDIT=False 
USE_CF_KE=False
QUAL_ANAL=False
EACH_BBOX=False
task_b=False

if KCC:
    INCLUDE_BBOX=False
    KNOWLEDGE_EDIT=True 
    EACH_BBOX=False
    task_b=False

# ----------------------------------------------------

def format_data(data, prompt=PROMPT, processor=None, model_name=MODEL_NAME):
    caption, image, image_path = data['caption'], data['image'], data['image_path']

    if hasattr(processor, "apply_chat_template"):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]
    elif "deepseek" in model_name.lower():
        return [{
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": [image_path] if isinstance(image_path, str) else [None],
            },
            {"role": "<|Assistant|>", "content": caption},
        ]
    else:
        return {"text": prompt, "image": image}

def transform_images(image_file, bbox_lst, type="blur"):
    image = Image.open(image_file).convert("RGB")
    if type == "blur":
        ret_image = blur_except_boxes(image, bbox_lst)
        ret_image = draw_bounding_boxes(ret_image, bbox_lst, each_bbox=EACH_BBOX)
    elif type == "crop":
        ret_image = crop_box(image, bbox_lst)

    if "7B" in MODEL_NAME:
        img_size = ret_image.size
        if img_size[0] > 900:
            img_size = (int(i * 0.8) for i in img_size)
            ret_image = ret_image.resize(img_size, Image.LANCZOS)

    return ret_image

def load_model(model_name: str, use_vllm=False):
    """모델 로드: vLLM 또는 HF Transformers"""
    if use_vllm:
        from vllm import LLM
        llm = LLM(model=model_name, tensor_parallel_size=1)
        return llm, None
    else:
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

def generate_text(model, processor, inputs, prompt, params):
    """통합 inference 함수"""
    # if USE_VLLM:
    #     # vLLM용 inference
    #     from vllm import SamplingParams
    #     sampling = SamplingParams(max_tokens=params.get("max_new_tokens", 128))
    #     results = model.generate(
    #         [f"{prompt}\n{data['caption']}" for data in inputs],
    #         sampling_params=sampling
    #     )
    #     return [r.outputs[0].text for r in results]
    # else:
    #     return generate_text_from_sample(model, processor, inputs, **params)

    # 1. InternVL 계열
    if hasattr(model, "chat"):
        outputs = []
        for data in inputs:
            image = data["content"][1]["image"]
            text = data["content"][1]["text"]
            out = model.chat(image, text)
            outputs.append(out)
        return outputs

    # 2. HuggingFace Vision2Seq 계열 (Qwen, LLaVA, Phi-3, etc.)
    elif hasattr(processor, "apply_chat_template"):
        return generate_text_from_sample(model, processor, inputs, **params)

    # 3. Idefics / DeepSeek / CogVLM 계열 (processor 있지만 chat 템플릿 다름)
    else:
        texts = [data["content"][1]["text"] for data in inputs]
        images = [data["content"][1]["image"] for data in inputs]
        model_inputs = processor(text=texts, images=images, return_tensors="pt").to(model.device)
        generated = model.generate(**model_inputs, max_new_tokens=params.get("max_new_tokens", 128))
        return processor.batch_decode(generated, skip_special_tokens=True)


def generate_text_from_sample(model, processor, samples, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    texts = [
        processor.apply_chat_template(sample[1:2], tokenize=False, add_generation_prompt=True)
        for sample in samples
    ]

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(samples)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens).to("cpu")

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    return trimmed_generated_ids


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading model: {MODEL_NAME}")

    model, processor = load_model(MODEL_NAME, use_vllm=USE_VLLM)
    if LOAD_ADAPTER and not USE_VLLM:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    ##################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Load data")
    directory_path = ROOT_PATH / CAPTION_PATH
    if QUAL_ANAL:
        with open(str(directory_path/"lace_test.json"), 'r') as f:
            eval_data = json.load(f)
            
        eval_lst = ['2363033', '2362969', '2362742', '2362682', '2341610', '2341436', '2341272', '2341163', '2341046', '2340962', '2340908', '2340907', '2340806', '2340762', '2340743', '2340711', '2340660', '2340484', '2340480', '2340476', '2340367', '2340303', '2340293', '2340230', '2340111', '2340104', '2340017', '2339863', '2339852', '2339759', '2339725', '2339715', '2339606', '2339537', '2339515', '2339510']
        eval_data = [{img_id:eval_data[img_id]} for img_id in eval_lst]
    else:
        if KCC:
            eval_data = get_each_json(sorted(glob.glob(str(directory_path / 'kcc/raw/*json'))))    # kcc 2025
        else:
            eval_data = get_each_json(sorted(glob.glob(str(directory_path / 'test/*json'))))  # general

    model_params = {}

    if KNOWLEDGE_EDIT:
        with open(str(directory_path/"train_keyword_dict.json"), 'r') as synf:
            tr_synset = json.load(synf)
            tr_synset = {k.split('.')[0]: v for k, v in tr_synset.items()}
        with open(str(directory_path/"test_keyword_dict.json"), 'r') as synf:
            te_synset = json.load(synf)
            te_synset = {k.split('.')[0]: v for k, v in te_synset.items()}

        def get_synlst(synset):
            syn_lst = []
            for _, vs in synset.items():
                vs = [v.lower() for v in vs]
                syn_lst.extend(vs)
            return syn_lst
        tr_synlst = get_synlst(tr_synset)
        te_synlst = get_synlst(te_synset)
        syn_lst = tr_synlst + te_synlst
        syn_lst = list(set(syn_lst))

        eval_data, eval_p_data = get_edit_examples(eval_data, eval=True, prompt=EDIT_PROMPT, use_cf=USE_CF_KE)
        model_params["max_new_tokens"]= 3
        objs = eval_data['objs']
    else:
        eval_data, eval_p_data = get_captions(eval_data, eval=True, prompt_c=CAP_PROMPT, prompt_b=PAP_PROMPT, include_bbox=INCLUDE_BBOX)
        PROMPT = CAP_PROMPT
        model_params["max_new_tokens"]= 64
    
    if task_b:
        eval_data['image_path'] = eval_p_data["image_path"]
        eval_data['caption'] = eval_p_data['paragraph']
        eval_data['bounding_box'] = eval_p_data['sub_region_boxes']
        # eval_data['bounding_box'] = eval_p_data['p_bounding_box']
        eval_data['candidates'] = [[ff] for ff in eval_p_data['paragraph']]

        model_params["max_new_tokens"]= 512
        PROMPT = PAP_PROMPT

        if KNOWLEDGE_EDIT:
            model_params["max_new_tokens"]= 5
            objs = eval_p_data['objs_in_paragraph']

    candidates = eval_data["candidates"]

    vl_lst = []
    crop_img_lst = []
    for c, ip, bb, pp in tqdm(zip(eval_data["caption"], eval_data["image_path"], eval_data["bounding_box"], eval_data["prompt"]), total=len(eval_data["caption"])):
        vl_lst.append({"image": transform_images(ip, bb), "caption": c, "prompt": pp, "image_path": ip})
        crop_img_lst.append(transform_images(ip, bb, "crop"))

    if QUAL_ANAL:
        img_ids = []
        img_id2 = ''
        for el_i, el in enumerate(eval_data['image_path']):
            img_id = el.split('/')[-1].split('.')[0]
            if img_id != img_id2: tmp_num=0
            else: tmp_num += 1
            img_new_id = f"{img_id}-{tmp_num}.jpg"
            vl_lst[el_i]['image'].save(f"qual_anal_results/{img_new_id}")
            img_ids.append(img_new_id)
            img_id2 = el.split('/')[-1].split('.')[0]

    if KNOWLEDGE_EDIT:
        formatted = [format_data(data, data['prompt'], processor, MODEL_NAME) for data in vl_lst]
    else:
        formatted = [format_data(data, PROMPT, processor, MODEL_NAME) for data in vl_lst]


    # check null candidate
    new_eval_data, new_candidates, new_crop_img_lst = [], [], []
    for cand, ed, cimg in tqdm(zip(candidates, formatted, crop_img_lst), total=len(formatted)):
        if cand != []:
            new_eval_data.append(ed)
            new_candidates.append(cand)
            new_crop_img_lst.append(cimg)
    eval_data = new_eval_data
    candidates = new_candidates
    crop_img_lst = new_crop_img_lst

    formatted_batches = [formatted[i:i+BATCH_SIZE] for i in range(0, len(formatted), BATCH_SIZE)]

    all_gen = []
    for batch in tqdm(formatted_batches, total=len(formatted_batches), desc="Run model"):
        all_gen.extend(generate_text(model, processor, batch, PROMPT, model_params))
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if KNOWLEDGE_EDIT:
        acc = compute_acc(all_gen, objs, syn_lst, te_synset, processor)
        print(f"Knowledge Editing Accuracy: {acc:.4f}")
    else:
        metric_ret, gen_seq = compute_metrics_custom(all_gen, candidates, crop_img_lst, processor)
        print(metric_ret)

if __name__ == "__main__":
    main()
