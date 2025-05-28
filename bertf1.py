# calculate bert f1 score for a given candidate and reference
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import bert_score
from bert_score import score
import PyPDF2

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

def extract_text_from_pdf(pdf_path):
    """Extracts and concatenates all text from a PDF file."""
    text = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text.strip())
    return ' '.join(text)

if __name__ == "__main__":
    candidate_folder = "./outputs/specs/"
    candidate_path = []
    for file in os.listdir(candidate_folder):
        if file.startswith("接地"):
            candidate_path.append(os.path.join(candidate_folder, file))
    reference_path = "./data_source/eval_demo/gpt.txt"
    cands = []
    for path in candidate_path:
        with open(path) as f:
            cand = [line.strip() for line in f]
            cand = " ".join(cand)
            cands.append(cand)
    with open(reference_path) as f:
        cands_gpt = [line.strip() for line in f]
    cands_gpt = [" ".join(cands_gpt)]
    # Use PDF as reference
    ref_pdf_path = "./data_source/eval_demo/16060 接地-B16.pdf"  # Change this to your actual PDF path
    refs = [extract_text_from_pdf(ref_pdf_path)]

    refs_cand = [refs for _ in range(len(cands))]
    assert len(cands) == len(refs_cand)
    P, R, F1 = score(cands, refs_cand, lang='zh', verbose=True)
    P_gpt, R_gpt, F1_gpt = score(cands_gpt, refs, lang='zh', verbose=True)
    P_baseline, R_baseline, F1_baseline = score(refs, refs, lang='zh', verbose=True)
    # f1 = calculate_bert_f1(cands, refs)
    # f1_gpt = calculate_bert_f1(cands_gpt, refs)
    # f1 = bert_score.score(cands, refs, lang="en", model_type="bert-base-uncased")
    print("P, R, F1: ", P, R, F1)
    print("P_gpt, R_gpt, F1_gpt: ", P_gpt, R_gpt, F1_gpt)
    print("P_baseline, R_baseline, F1_baseline: ", P_baseline, R_baseline, F1_baseline)
