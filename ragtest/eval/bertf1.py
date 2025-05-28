# calculate bert f1 score for a given candidate and reference

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import bert_score
from bert_score import score

def calculate_bert_f1(candidate, reference):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    candidate_tokens = tokenizer.encode(candidate, add_special_tokens=True)
    reference_tokens = tokenizer.encode(reference, add_special_tokens=True)

    candidate_tokens = torch.tensor(candidate_tokens).unsqueeze(0)
    reference_tokens = torch.tensor(reference_tokens).unsqueeze(0)

    with torch.no_grad():
        outputs = model(candidate_tokens, reference_tokens)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    return preds

if __name__ == "__main__":
    with open("hyps.txt") as f:
        cands = [line.strip() for line in f]
    # concat cands
    cands = [" ".join(cands)]

    with open("refs.txt") as f:
        refs = [line.strip() for line in f]
    # concat refs
    refs = [" ".join(refs)]

    P, R, F1 = score(cands, refs, lang='zh', verbose=True)
    # f1 = calculate_bert_f1(cands, refs)
    # f1 = bert_score.score(cands, refs, lang="en", model_type="bert-base-uncased")
    print(P, R, F1)
