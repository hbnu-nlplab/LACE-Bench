import os
from tqdm import tqdm
from PIL import Image

from PIL import Image, ImageFilter, ImageDraw
import json

from . import ROOT_PATH, IMG_PATH
colors = ["green", "red", "blue", "yellow", "orange", "purple", "white", "brown", "pink"]


def blur_except_boxes(image, bounding_boxes):
    """
    Apply blur to all areas except the selected bounding box.

    :param image_path: Path to the input image.
    :param bounding_boxes: List of bounding boxes in the format [(x1, y1, x2, y2), ...].
    :param output_path: Path to save the output image.
    """
    # Apply blur filter to the entire image
    blurred_image = image.filter(ImageFilter.GaussianBlur(15))

    # Create a mask for the unblurred area
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    x, y, w, h = 999, 999, 0, 0

    for box in bounding_boxes:
        bx, by, bw, bh = box
        if bx < x:
            x = bx
        if by < y:
            y = by
        if bw > w:
            w = bw
        if bh > h:
            h = bh
        # int_box = tuple(map(int, box))
    bbox = [x, y, w, h]
    draw.rectangle(bbox, fill=255)

    # Composite the images using the mask
    final_image = Image.composite(image, blurred_image, mask)

    return final_image



def draw_bounding_boxes(image, bounding_boxes, each_bbox=False, outline="green"):
    x, y, w, h = 999, 999, 0, 0
    draw = ImageDraw.Draw(image)
    for bi, box in enumerate(bounding_boxes):
        bx, by, bw, bh = box

        if each_bbox:
            draw.rectangle([bx, by, bw, bh], outline=colors[bi], width=2)
        else:
            bx, by, bw, bh = box
            if bx < x:
                x = bx
            if by < y:
                y = by
            if bw > w:
                w = bw
            if bh > h:
                h = bh

    if not each_bbox:
        bbox = [x, y, w, h]
        draw.rectangle(bbox, outline='green', width=2)

    return image
    

def crop_box(image, bounding_box):
    if type(bounding_box) == list: bounding_box = bounding_box[0]
    cropped_image = image.crop(bounding_box)
    
    return cropped_image

def get_each_json(ann_paths):
        annotation = []
        for ann_path in ann_paths:
            data = json.load(open(ann_path, "r"))
            if type(data) == dict:
                annotation.append(data)
        return annotation



def get_captions(annotation, eval=False, prompt_c="", prompt_b="", include_bbox=False, counterfactual=False):
    vis_root = str(ROOT_PATH / IMG_PATH)
    if counterfactual: capkey = "counterfactual_caption"
    else: capkey = "caption"
    # data = []
    data = {'image_path':[], 'caption':[], 'prompt':[], 'bounding_box': [], "candidates": []}
    p_data = {'image_path': [], 'paragraph': [], 'sub_regions': [], 'sub_region_boxes': [], 'p_bounding_box': [], 'prompt': []}
    for i, record in tqdm(enumerate(annotation), total=len(annotation)):
        image_id = next(iter(record.keys()))
        image_path = os.path.join(vis_root, image_id+".jpg")
        
        # for blured_image
        for region in record[image_id]['regions']:
            bbox = (region['x'], region['y'], region['x']+region['width'], region['y']+region['height'])
                    
            if eval:
                captions = region['captions'][:1]
            else:
                captions = region['captions']

            for cap_dict in captions:
                if prompt_c:
                    prompt = prompt_c
                else:
                    prompt = "Describe a sentence for the given image."
                if include_bbox:
                    prompt = prompt + f" Refer to the position of the bounding box, which is {bbox}."
                else:
                    prompt = prompt
                
                data['prompt'].append(prompt)
                data['image_path'].append(image_path)
                data['caption'].append(cap_dict[capkey])
                data['bounding_box'].append([bbox])

            data['candidates'].append([cap_dict[capkey] for cap_dict in region['captions']])

        for rr in record[image_id]["relation_centric_regions"]:
            x, y, w, h = 999, 999, 0, 0
            err_flag = False
            tmp_region_boxes = []
            if len(rr['region_ids']) == 0: print(image_id); continue
            for rid in rr['region_ids']:
                try:
                    rid = int(rid.split('_')[-1].replace(',', ''))
                except: 
                    print(rid)
                    err_flag = True
                    continue

                try:
                    b_x = record[image_id]['regions'][rid]['x']
                    b_w = record[image_id]['regions'][rid]['width']+record[image_id]['regions'][rid]['x']
                    b_y = record[image_id]['regions'][rid]['y']
                    b_h = record[image_id]['regions'][rid]['height']+record[image_id]['regions'][rid]['y']

                    if b_x < x:
                        x = b_x
                    if b_y < y:
                        y = b_y
                    if b_w > w:
                        w = b_w
                    if b_h > h:
                        h = b_h

                    tmp_region_boxes.append([b_x, b_y, b_w, b_h])
                except:
                    print(rid, )
                    err_flag = True
                    continue

            if not err_flag:
                p_data['sub_region_boxes'].append(tmp_region_boxes)
                p_bbox = (x, y, x+w, y+h)
                p_data['image_path'].append(image_path)
                p_data['paragraph'].append(rr['human_annotation'])
                p_data['sub_regions'].append(rr['region_ids'])
                p_data['p_bounding_box'].append(p_bbox)

                if prompt_b:
                    prompt = prompt_b
                else:
                    prompt = "The given image defines several objects as a group and creates a bounding box. Describe this bounding box at paragraph level. Paragraphs must consist of at least three sentences."
                if include_bbox:
                    prompt = prompt + f" Refer to the position of the bounding box, which is {p_bbox}."
                else:
                    prompt = prompt

                p_data['prompt'].append(prompt)
    return data, p_data


def get_objs(caption, cf_caption):
    tokens_caption = caption.split()
    tokens_cf = cf_caption.split()
    
    # 1. 공통 prefix 찾기
    prefix_len = 0
    for w1, w2 in zip(tokens_caption, tokens_cf):
        if w1 == w2:
            prefix_len += 1
        else:
            break

    # 2. 공통 suffix 찾기
    suffix_len = 0
    # 남은 단어가 없지 않도록 조건을 걸어줌
    while (suffix_len < len(tokens_caption) - prefix_len and 
           suffix_len < len(tokens_cf) - prefix_len and 
           tokens_caption[-(suffix_len+1)] == tokens_cf[-(suffix_len+1)]):
        suffix_len += 1

    # 3. 차이나는 부분(중간 부분) 추출
    # suffix_len이 0이면 tokens_caption[-0:] 는 전체가 되므로 따로 처리
    if suffix_len > 0:
        original_tokens = tokens_caption[prefix_len: -suffix_len]
        masked_caption = tokens_caption[:prefix_len] + [" {object} "] + tokens_caption[-suffix_len:]
        cf_tokens = tokens_cf[prefix_len: -suffix_len]
    else:
        original_tokens = tokens_caption[prefix_len:]
        masked_caption = tokens_caption[:prefix_len] + [" {object} "]
        cf_tokens = tokens_cf[prefix_len:]
    
    # 토큰을 다시 문자열로 합치기
    original_word = " ".join(original_tokens)
    cf_word = " ".join(cf_tokens)
    masked_caption = " ".join(masked_caption)
    
    return original_word, cf_word, masked_caption



def get_edit_examples(annotation, eval=False, prompt="", use_cf=False):
    vis_root = str(ROOT_PATH / IMG_PATH)
    # data = []
    data = {'image_path':[], 'caption':[], 'prompt':[], 'bounding_box': [], "candidates": [], "objs": [], "counterfactual_objs": []}
    p_data = {'image_path': [], 'paragraph': [], 'sub_regions': [], 'sub_region_boxes': [], 'p_bounding_box': [], 'prompt': [], 'objs_in_paragraph': []}
    for i, record in tqdm(enumerate(annotation), total=len(annotation)):
        image_id = next(iter(record.keys()))
        image_path = os.path.join(vis_root, image_id+".jpg")
        
        obj_lst = []
        # for blured_image
        for region in record[image_id]['regions']:
            bbox = (region['x'], region['y'], region['x']+region['width'], region['y']+region['height'])
                    
            if eval:
                captions = region['captions'][:1]
            else:
                captions = region['captions']

            for cap_dict in captions:
                data['image_path'].append(image_path)
                caption = cap_dict['caption']
                counterfactual_caption = cap_dict['counterfactual_caption']
                original_obj, counterfact_obj, masked_caption = get_objs(caption, counterfactual_caption)
                if use_cf:
                    obj_lst.append(counterfact_obj)
                    data['objs'].append(counterfact_obj)
                    data['counterfactual_objs'].append(counterfact_obj)
                    data['candidates'].append([counterfactual_caption])
                else:
                    obj_lst.append(original_obj)
                    data['objs'].append(original_obj)
                    data['counterfactual_objs'].append(counterfact_obj)
                    data['candidates'].append([caption])

                data['prompt'].append(prompt + '\n[CAPTION] ' + masked_caption)
                data['caption'].append(masked_caption)
                data['bounding_box'].append([bbox])

            # data['candidates'].append([cap_dict['caption'] for cap_dict in region['captions']])

        for rr in record[image_id]["relation_centric_regions"]:
            x, y, w, h = 999, 999, 0, 0
            err_flag = False
            tmp_region_boxes = []
            for rid in rr['region_ids']:
                try:
                    rid = int(rid.split('_')[-1].replace(',', ''))
                except: 
                    print(rid)
                    err_flag = True
                    continue

                try:
                    b_x = record[image_id]['regions'][rid]['x']
                    b_w = record[image_id]['regions'][rid]['width']+record[image_id]['regions'][rid]['x']
                    b_y = record[image_id]['regions'][rid]['y']
                    b_h = record[image_id]['regions'][rid]['height']+record[image_id]['regions'][rid]['y']

                    if b_x < x:
                        x = b_x
                    if b_y < y:
                        y = b_y
                    if b_w > w:
                        w = b_w
                    if b_h > h:
                        h = b_h

                    tmp_region_boxes.append([b_x, b_y, b_w, b_h])                
                except:
                    print(rid, )
                    err_flag = True
                    continue

            if not err_flag:
                paragraph = rr['human_annotation']

                for obj in obj_lst:
                    obj_b_idx = paragraph.find(obj)
                    if obj_b_idx != -1:
                        objs_in_paragraph = paragraph[obj_b_idx: obj_b_idx+len(obj)]
                        new_paragraph = paragraph[:obj_b_idx] + "{object}" + paragraph[obj_b_idx+len(obj):]

                        p_data['sub_region_boxes'].append(tmp_region_boxes)
                        p_data['image_path'].append(image_path)
                        p_data['paragraph'].append(new_paragraph)
                        p_data['objs_in_paragraph'].append(objs_in_paragraph)
                        p_data['sub_regions'].append(rr['region_ids'])
                        p_data['p_bounding_box'].append((x, y, x+w, y+h))

                        if prompt:
                            p_data['prompt'].append(prompt)
                        else:
                            p_data['prompt'].append("The given image defines several objects as a group and creates a bounding box. Describe this bounding box at paragraph level. Paragraphs must consist of at least three sentences.")
    return data, p_data