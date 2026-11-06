"""Zero-shot CLIP or SigLIP scores per photo (experiment, not part of the pipeline).

For each question the score is the image's logit for the averaged "problem" prompts minus the logit
for the averaged "fine" prompts; higher means more likely a problem. Run it in its own environment,
since it needs PyTorch and open_clip, which the pipeline doesn't (outside the repository, about 1 GB):

    uv venv ~/.venvs/clip && VIRTUAL_ENV=~/.venvs/clip uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=~/.venvs/clip uv pip install open_clip_torch transformers sentencepiece protobuf pandas
    ~/.venvs/clip/bin/python scripts/clip_zero_shot.py inputs.csv scores.csv [model pretrained]

model and pretrained are open_clip names (open_clip.list_pretrained()); the default is ViT-B-16-SigLIP webli.

inputs.csv has the columns path, set, file and an optional eye box x0, y0, x1, y1 (pixels); the box used
for the Open Images test was the eye centre +-1.1 x the eye distance wide and +-0.55 x high, from OFIQ's
landmarks. The prompts were written for ViT-B-16-SigLIP and not tuned per model.

AUC against the labels on the Open Images set (2026-11-06; notebooks/real_photos.ipynb, section 4):

    question      ours  SigLIP B-16  CLIP B-32           CLIP L-14         SigLIP SO400M
                        (800 MB)     (LAION, 600 MB)     (OpenAI, 1.6 GB)  (3.3 GB)
    glare         0.56  0.78         0.74                0.82              0.81
    tinted        0.96  0.98         0.98                0.99              0.99
    hat           0.94  0.94         0.88                0.96              0.96
    eyes_closed   0.83  0.80         0.79                0.90              0.84
    gaze_aside    0.76  0.71         0.53                0.63              0.83

open_clip names: ViT-B-16-SigLIP webli, ViT-B-32 laion2b_s34b_b79k, ViT-L-14 openai, ViT-SO400M-14-SigLIP webli.
"""
import sys

import open_clip
import pandas as pd
import torch
from PIL import Image

INPUTS, OUT = sys.argv[1], sys.argv[2]
MODEL, PRETRAINED = sys.argv[3:5] if len(sys.argv) > 4 else ('ViT-B-16-SigLIP', 'webli')

# question: (region, problem prompts, fine prompts)
QUESTIONS = {
    'glare': ('eyes', [
        'eyeglasses with a bright glare reflection on the lenses',
        'glasses lenses reflecting light, with a white shine across the glass',
        'a reflection of a window or lamp visible on the eyeglass lenses',
    ], [
        'eyeglasses with clean clear lenses and no reflections',
        'glasses through which the eyes are clearly visible, without any glare',
        'clear eyeglass lenses without any reflection',
    ]),
    'tinted': ('face', [
        'a photo of a person wearing sunglasses',
        'a photo of a person wearing tinted glasses that hide the eyes',
        'a photo of a person wearing dark or mirrored lenses',
    ], [
        'a photo of a person wearing clear eyeglasses',
        'a photo of a person without glasses',
        'a photo of a person whose eyes are clearly visible',
    ]),
    'hat': ('face', [
        'a photo of a person wearing a hat',
        'a photo of a person wearing a cap or beanie',
        'a photo of a person with a head covering such as a scarf, hood or headband',
    ], [
        'a photo of a person with an uncovered head',
        'a photo of a person with bare hair and no hat',
        'a photo of a bald person without a hat',
    ]),
    'eyes_closed': ('eyes', [
        'a close-up of closed eyes',
        'a close-up of eyes blinking or squeezed shut',
        'a close-up of eyes looking down with lowered eyelids',
    ], [
        'a close-up of wide open eyes',
        'a close-up of open eyes looking at the camera',
        'a close-up of eyes that are clearly open',
    ]),
    'gaze_aside': ('eyes', [
        'a close-up of eyes looking to the side',
        'a close-up of eyes looking away from the camera',
        'a close-up of eyes looking up or down',
    ], [
        'a close-up of eyes looking straight into the camera',
        'a close-up of eyes making direct eye contact with the viewer',
        'a close-up of eyes gazing directly at the lens',
    ]),
    'filter': ('face', [
        'a selfie with a Snapchat filter',
        'a photo with cartoon stickers, animal ears or face effects added',
        'a heavily edited photo with a beauty filter and airbrushed skin',
    ], [
        'an unedited photo of a person',
        'a natural photo of a face without any filters or effects',
        'a plain passport photo',
    ]),
}

model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL)
model.eval()
torch.set_num_threads(8)


def text_embedding(prompts):
    with torch.no_grad():
        e = model.encode_text(tokenizer(prompts))
    e = e / e.norm(dim=-1, keepdim=True)
    e = e.mean(0)
    return e / e.norm()


texts = {q: (region, text_embedding(pos), text_embedding(neg)) for q, (region, pos, neg) in QUESTIONS.items()}
scale = model.logit_scale.exp().item()

rows = []
for _, row in pd.read_csv(INPUTS).iterrows():
    image = Image.open(row['path']).convert('RGB')
    crops = {'face': image}
    if pd.notna(row.get('x0')):
        w, h = image.size
        box = (max(0, int(row['x0'])), max(0, int(row['y0'])), min(w, int(row['x1'])), min(h, int(row['y1'])))
        if box[2] - box[0] > 10 and box[3] - box[1] > 10:
            crops['eyes'] = image.crop(box)
    with torch.no_grad():
        embeddings = {}
        for name, crop in crops.items():
            e = model.encode_image(preprocess(crop).unsqueeze(0))[0]
            embeddings[name] = e / e.norm()
    out = dict(file=row['file'], set=row['set'])
    for q, (region, pos, neg) in texts.items():
        if region in embeddings:
            e = embeddings[region]
            out[q] = scale * float(e @ pos) - scale * float(e @ neg)  # SigLIP's logit bias cancels out
    rows.append(out)
    if len(rows) % 100 == 0:
        print(len(rows), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print('done', len(rows))
