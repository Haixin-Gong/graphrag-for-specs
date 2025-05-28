import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from bert_score import score
import jieba
from sentence_transformers import SentenceTransformer
import torch
import PyPDF2
import re
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba.analyse

class TextEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.smoothing = SmoothingFunction().method1
        self.technical_terms = set()
        
    def build_technical_terms(self, reference_text, corpus_texts, top_n=50):
        """构建技术术语词典
        
        Args:
            reference_text: 参考文档文本
            corpus_texts: 文档库文本列表
            top_n: 选择前N个最重要的词作为技术术语
        """
        # 使用jieba的TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(
            reference_text,
            topK=top_n,
            withWeight=True,
            allowPOS=('n', 'vn', 'v', 'a', 'an')  # 只考虑名词、动名词、动词、形容词
        )
        
        # 获取参考文档中的词频
        reference_words = list(jieba.cut(reference_text))
        reference_word_freq = Counter(reference_words)
        
        # 计算文档库中的词频
        corpus_word_freq = Counter()
        for text in corpus_texts:
            corpus_word_freq.update(jieba.cut(text))
        
        # 计算词的TF-IDF分数
        word_scores = {}
        for word, weight in keywords:
            # TF: 词在参考文档中的频率
            tf = reference_word_freq[word] / len(reference_words)
            # IDF: 词在文档库中的逆频率
            idf = math.log(len(corpus_texts) / (1 + corpus_word_freq[word]))
            # 综合分数
            word_scores[word] = tf * idf * weight
        
        # 选择分数最高的词作为技术术语
        self.technical_terms = set(dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]).keys())
        
        return self.technical_terms
        
    def calculate_technical_metrics(self, candidate, reference):
        """计算技术文档特定的评估指标"""
        # 提取数字和单位
        def extract_numbers_and_units(text):
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            units = re.findall(r'(?:mm|cm|m|kg|g|V|A|W|Hz|Ω|°C|%|s|min|h)', text)
            return set(numbers), set(units)
        
        # 使用jieba分词
        candidate_words = list(jieba.cut(candidate))
        reference_words = list(jieba.cut(reference))
        
        # 计算技术术语的覆盖率
        candidate_terms = set(candidate_words).intersection(self.technical_terms)
        reference_terms = set(reference_words).intersection(self.technical_terms)
        
        term_coverage = len(candidate_terms) / len(reference_terms) if reference_terms else 0
        
        # 计算数字和单位的匹配率
        candidate_numbers, candidate_units = extract_numbers_and_units(candidate)
        reference_numbers, reference_units = extract_numbers_and_units(reference)
        
        number_match = len(candidate_numbers.intersection(reference_numbers)) / len(reference_numbers) if reference_numbers else 0
        unit_match = len(candidate_units.intersection(reference_units)) / len(reference_units) if reference_units else 0
        
        # 计算技术术语的权重得分
        term_scores = {}
        for term in self.technical_terms:
            if term in candidate_words and term in reference_words:
                # 计算词在参考文档中的权重
                ref_weight = reference_words.count(term) / len(reference_words)
                # 计算词在候选文档中的权重
                cand_weight = candidate_words.count(term) / len(candidate_words)
                # 计算权重差异
                term_scores[term] = 1 - abs(ref_weight - cand_weight)
        
        weighted_term_score = sum(term_scores.values()) / len(term_scores) if term_scores else 0
        
        return {
            'term_coverage': term_coverage,
            'weighted_term_score': weighted_term_score,
            'number_match': number_match,
            'unit_match': unit_match,
            'technical_terms': list(self.technical_terms)  # 返回识别出的技术术语列表
        }
    
    def calculate_rouge(self, candidate, reference):
        """计算ROUGE分数"""
        # 使用jieba进行分词
        candidate_words = ' '.join(jieba.cut(candidate))
        reference_words = ' '.join(jieba.cut(reference))
        
        # 计算ROUGE分数
        scores = self.rouge.get_scores(candidate_words, reference_words)
        
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f']
        }
    
    def calculate_bleu(self, candidate, reference):
        """计算BLEU分数"""
        # 对中文文本进行分词
        candidate_tokens = list(jieba.cut(candidate))
        reference_tokens = list(jieba.cut(reference))
        
        # 计算不同n-gram的BLEU分数
        weights = [(1, 0, 0, 0),  # 1-gram
                  (0.5, 0.5, 0, 0),  # 2-gram
                  (0.33, 0.33, 0.33, 0),  # 3-gram
                  (0.25, 0.25, 0.25, 0.25)]  # 4-gram
        
        bleu_scores = {}
        for i, weight in enumerate(weights, 1):
            bleu_scores[f'bleu-{i}'] = sentence_bleu(
                [reference_tokens], 
                candidate_tokens,
                weights=weight,
                smoothing_function=self.smoothing
            )
        
        return bleu_scores
    
    def calculate_semantic_similarity(self, candidate, reference):
        """计算语义相似度"""
        # 使用Sentence-BERT计算语义相似度
        embeddings1 = self.sentence_model.encode(candidate, convert_to_tensor=True)
        embeddings2 = self.sentence_model.encode(reference, convert_to_tensor=True)

        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.unsqueeze(0)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.unsqueeze(0)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        return similarity.item()

    def calculate_bert_f1(self, candidate, reference):
        """计算BERT F1分数"""
        P, R, F1 = score([candidate], [reference], lang='zh', verbose=False)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    
    def evaluate_all(self, candidate, reference):
        """计算所有评估指标"""
        results = {
            'technical_metrics': self.calculate_technical_metrics(candidate, reference),
            'semantic_similarity': self.calculate_semantic_similarity(candidate, reference),
            'bert_f1': self.calculate_bert_f1(candidate, reference),
            'rouge': self.calculate_rouge(candidate, reference),
            'bleu': self.calculate_bleu(candidate, reference)
        }
        return results

def process_pdf(pdf_file):
    """处理PDF文件"""
    # 使用PyPDF2读取PDF文件
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# 使用示例
if __name__ == "__main__":
    evaluator = TextEvaluator()
    
    # 读取文件
    cand_file_1 = "./outputs/specs/接地_global_20250520_152953_specs.txt"
    cand_file_2 = "./outputs/specs/接地_Local Knowledge Base_20250520_153555_specs.txt"
    cand_file_3 = "./outputs/specs/接地_unknown_20250520_152541_specs.txt"
    cand_file_4 = "./outputs/specs/接地_User-defined Knowledge Base_20250520_153818_specs.txt"
    gpt_file = "./data_source/eval_demo/gpt.txt"
    ref_file = "./data_source/eval_demo/16060 接地-B16.pdf"

    # 读取文件内容
    candidate_1 = open(cand_file_1, "r").read()
    candidate_2 = open(cand_file_2, "r").read()
    candidate_3 = open(cand_file_3, "r").read()
    candidate_4 = open(cand_file_4, "r").read()
    baseline = open(gpt_file, "r").read()
    reference = process_pdf(ref_file)

    # 构建技术术语词典
    corpus_texts = [candidate_1, candidate_2, candidate_3, candidate_4]
    technical_terms = evaluator.build_technical_terms(reference, corpus_texts)
    print("\n识别出的技术术语:")
    print(technical_terms)

    # 评估所有候选文本
    candidates = {
        "candidate_1": candidate_1,
        "candidate_2": candidate_2,
        "candidate_3": candidate_3,
        "candidate_4": candidate_4,
        "baseline": baseline
    }

    for name, candidate in candidates.items():
        results = evaluator.evaluate_all(candidate, reference)
        print(f"\n{name} 评估结果:")
        print(f"技术指标:")
        print(f"  - 技术术语覆盖率: {results['technical_metrics']['term_coverage']:.4f}")
        print(f"  - 技术术语权重得分: {results['technical_metrics']['weighted_term_score']:.4f}")
        print(f"  - 数值匹配率: {results['technical_metrics']['number_match']:.4f}")
        print(f"  - 单位匹配率: {results['technical_metrics']['unit_match']:.4f}")
        print(f"语义相似度: {results['semantic_similarity']:.4f}")
        print(f"BERT F1分数: {results['bert_f1']}")
        print(f"ROUGE分数: {results['rouge']}")
        print(f"BLEU分数: {results['bleu']}")
        print("-" * 50)