import numpy as np
from statistics import mean
import evaluate
from torchmetrics.multimodal.clip_score import CLIPScore
import torch, torchvision
from nltk.tokenize import word_tokenize  
from nltk.tag import pos_tag  

print("LOAD METRIC ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = evaluate.combine(["rouge", "meteor"])
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
pil_to_tensor = torchvision.transforms.PILToTensor()
print("done.")


def calc_bertscore(decoded_preds, candidates_n, lang, device):
    scores = []
    for pred, refs in zip(decoded_preds, candidates_n):
        # 각 prediction을 n개의 candidate와 비교
        bs = bertscore.compute(
            predictions=[pred] * len(refs),
            references=refs,
            lang=lang,
            device=device,
        )
        # 각 reference별 F1 평균 계산
        f1_mean = mean(bs["f1"])
        scores.append(f1_mean)
    return scores

def compute_metrics_custom(preds, candidates, images, tokenizer, lang="en"):
    score_lst = ["rouge1", "rouge2", "rougeL", "meteor"]

    # ====== 0. Decode if tokenized / tensor ======
    if hasattr(preds, "device") or hasattr(preds, "cpu"):  # torch tensor
        preds = preds.cpu().numpy().tolist()
    if tokenizer is not None:
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    else:
        decoded_preds = preds

    # ====== 1. Text cleanup ======
    decoded_preds = [
        p.replace("The image features ", "")
         .replace("The image shows ", "")
         .replace("\n", " ")
         .strip()
        for p in decoded_preds
    ]

    # 평가 지표 계산n
    if type(decoded_preds) == str:
        decoded_preds = [decoded_preds]

    ret = {}

    # n = 5
    n=5
    candidates_n = [cands[:n] for cands in candidates]
    metric.add_batch(predictions=decoded_preds, references=candidates_n)
    # bleu
    eval_metric = metric.compute()
    eval_metric.update(bleu.compute(predictions=decoded_preds, references=candidates_n))
    for bleu_n in [1,2,3]:
        eval_metric[f'bleu-{bleu_n}'] = bleu.compute(predictions=decoded_preds, references=candidates_n, max_order=bleu_n)['bleu']


    bs = calc_bertscore(decoded_preds, candidates_n, lang, device=device)
    eval_metric["bertscore_f1"] = float(mean(bs)) 

    eval_metric.update({k:float(eval_metric[k]) for k in score_lst})

    ret[f"n=5"] = {k: float(v) for k, v in eval_metric.items() if isinstance(v, (int, float, np.floating))}

    # all
    metric.add_batch(predictions=decoded_preds, references=candidates)
    eval_metric = metric.compute()
    eval_metric.update(bleu.compute(predictions=decoded_preds, references=candidates))
    # bleu
    for bleu_n in [1,2,3]:
        eval_metric[f'bleu-{bleu_n}'] = bleu.compute(predictions=decoded_preds, references=candidates, max_order=bleu_n)['bleu']
    
    eval_metric.update({k:float(eval_metric[k]) for k in score_lst})

    # bertscore
    bs_all = calc_bertscore(decoded_preds, candidates, lang, device=device)
    eval_metric["bertscore_f1"] = float(mean(bs_all)) 

    # clipscore
    try:
        with torch.no_grad():
            scores = [clip_metric(pil_to_tensor(img), p) for img, p in zip(images, decoded_preds)]
        eval_metric["clipscore"] = float(torch.stack(scores).mean().item())/100
    except Exception as e:
        eval_metric["clipscore"] = float("nan")
        print(f"[WARN] CLIPScore computation failed: {e}")

    ret["n=all"] = {k: float(v) for k, v in eval_metric.items() if isinstance(v, (int, float, np.floating))}

    return ret, decoded_preds


def compute_acc(gen_batch, objs, syn_lst, te_synset, processor):
    decoded_preds = processor.batch_decode(gen_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    gen_seq = []
    for pi, pred in enumerate(decoded_preds):
        tokens = word_tokenize(pred)
        tagged = pos_tag(tokens)
        pred = ' '.join([word for word, pos in tagged if pos.startswith('N')])
        gen_seq.append(pred)

    acc = 0
    for pi, pred in enumerate(gen_seq):
        acc_flag = False
        if pred == objs[pi]:
            acc += 1
            acc_flag = True
        elif pred in syn_lst:
            if objs[pi] in te_synset:
                if pred in te_synset[objs[pi]]:
                    acc += 0.8
                    acc_flag = True
        elif pred == '':
            continue

        if not acc_flag:
            bs = bertscore.compute(predictions=[pred], references=[objs[pi]], lang="en", device=device)
            if bs['f1'][0] < 0.85:
                acc += bs['f1'][0] #* 0.05

    return acc