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
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler, RobustScaler

plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'  # 替换为你选择的字体

def normalize_metrics(df, metrics, method='minmax'):
    """
    对指标进行归一化处理，支持多种标准化方法
    
    Args:
        df: 包含原始指标的DataFrame
        metrics: 需要归一化的指标列表
        method: 标准化方法，可选值：
               - 'minmax': 最小-最大标准化 (x - min) / (max - min)
               - 'zscore': Z-score标准化 (x - mean) / std
               - 'robust': 鲁棒标准化，使用四分位数
               - 'maxabs': 最大绝对值标准化 x / |max|
               - 'log': 对数标准化 log(1 + x)
    
    Returns:
        normalized_df: 归一化后的DataFrame
    """
    normalized_df = df.copy()
    
    for metric in metrics:
        values = df[metric].values.reshape(-1, 1)
        
        if method == 'minmax':
            if np.max(values) - np.min(values) > 0:
                normalized_df[metric] = (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                normalized_df[metric] = values
        
        elif method == 'zscore':
            if np.std(values) > 0:
                normalized_df[metric] = (values - np.mean(values)) / np.std(values)
                # 将z-score映射到[0,1]区间，便于可视化
                zscore_values = normalized_df[metric].values
                normalized_df[metric] = (zscore_values - np.min(zscore_values)) / (np.max(zscore_values) - np.min(zscore_values))
            else:
                normalized_df[metric] = values
        
        elif method == 'robust':
            scaler = RobustScaler()
            normalized_values = scaler.fit_transform(values)
            # 将鲁棒标准化的结果映射到[0,1]区间
            normalized_df[metric] = (normalized_values - np.min(normalized_values)) / (np.max(normalized_values) - np.min(normalized_values))
        
        elif method == 'maxabs':
            max_abs = np.max(np.abs(values))
            if max_abs > 0:
                normalized_df[metric] = values / max_abs
            else:
                normalized_df[metric] = values
        
        elif method == 'log':
            # 对于可能包含0的数据，使用log(1+x)
            normalized_df[metric] = np.log1p(values)
            # 映射到[0,1]区间
            log_values = normalized_df[metric].values
            normalized_df[metric] = (log_values - np.min(log_values)) / (np.max(log_values) - np.min(log_values))
    
    return normalized_df

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
    # 创建一个文件处理器，只写入评估结果
    result_file = 'eval_demo/evaluation_metrics.txt'
    excel_file = 'eval_demo/evaluation_results.xlsx'
    radar_file = 'eval_demo/radar_plot.png'
    normalized_excel_file = 'eval_demo/normalized_evaluation_results.xlsx'

    evaluator = TextEvaluator()
    
    # 读取文件
    cand_file_1 = "./outputs/graghrag.txt"    
    cand_file_2 = "./outputs/globalrag.txt"
    cand_file_3 = "./outputs/globalrag2.txt"
    cand_file_4 = "./outputs/specsrag.txt"    
    ref_file = "./eval_demo/16060 接地-B16.pdf"

    # 读取文件内容
    candidate_1 = open(cand_file_1, "r").read()
    candidate_2 = open(cand_file_2, "r").read()
    candidate_3 = open(cand_file_3, "r").read()
    candidate_4 = open(cand_file_4, "r").read()
    reference = process_pdf(ref_file)

    # 构建技术术语词典
    corpus_texts = [candidate_1, candidate_2, candidate_3, candidate_4]
    technical_terms = evaluator.build_technical_terms(reference, corpus_texts)

    # 评估所有候选文本
    candidates = {
        "GraphRAG": candidate_1,
        "GlobalRAG": candidate_2,
        "GlobalRAG2": candidate_3,
        "SpecsRAG": candidate_4,
    }

    # 定义评估指标
    key_metrics = {
        "term_coverage": "技术术语覆盖率",
        "weighted_term_score": "技术术语权重得分",
        "number_match": "数值匹配率",
        "unit_match": "单位匹配率",
        "semantic_similarity": "语义相似度",
        "bert_f1": "BERT F1分数",
        "rouge-1": "ROUGE-1分数",
        "rouge-2": "ROUGE-2分数",
        "rouge-l": "ROUGE-L分数",
        "bleu-1": "BLEU-1分数",
        "bleu-2": "BLEU-2分数",
        "bleu-3": "BLEU-3分数",
        "bleu-4": "BLEU-4分数"
    }

    # 创建结果存储字典
    results_dict = {metric: [] for metric in key_metrics.keys()}
    results_dict['Model'] = []  # 添加模型名称列

    # 评估并收集结果
    with open(result_file, 'w') as f:
        f.write("识别出的技术术语:\n")
        f.write(str(technical_terms) + "\n\n")

        for name, candidate in candidates.items():
            results = evaluator.evaluate_all(candidate, reference)
            results_dict['Model'].append(name)
            
            # 写入文本结果
            f.write(f"{name} 评估结果:\n")
            f.write("技术指标:\n")
            
            # 收集技术指标
            tech_metrics = results['technical_metrics']
            results_dict['term_coverage'].append(tech_metrics['term_coverage'])
            results_dict['weighted_term_score'].append(tech_metrics['weighted_term_score'])
            results_dict['number_match'].append(tech_metrics['number_match'])
            results_dict['unit_match'].append(tech_metrics['unit_match'])
            
            # 收集语义相似度
            results_dict['semantic_similarity'].append(results['semantic_similarity'])
            
            # 收集BERT F1
            bert_p, bert_r, bert_f1 = results['bert_f1']
            results_dict['bert_f1'].append(bert_f1)
            
            # 收集ROUGE分数
            rouge_scores = results['rouge']
            results_dict['rouge-1'].append(rouge_scores['rouge-1'])
            results_dict['rouge-2'].append(rouge_scores['rouge-2'])
            results_dict['rouge-l'].append(rouge_scores['rouge-l'])
            
            # 收集BLEU分数
            bleu_scores = results['bleu']
            results_dict['bleu-1'].append(bleu_scores['bleu-1'])
            results_dict['bleu-2'].append(bleu_scores['bleu-2'])
            results_dict['bleu-3'].append(bleu_scores['bleu-3'])
            results_dict['bleu-4'].append(bleu_scores['bleu-4'])
            
            # 写入详细结果到文本文件
            f.write(f"  - 技术术语覆盖率: {tech_metrics['term_coverage']:.4f}\n")
            f.write(f"  - 技术术语权重得分: {tech_metrics['weighted_term_score']:.4f}\n")
            f.write(f"  - 数值匹配率: {tech_metrics['number_match']:.4f}\n")
            f.write(f"  - 单位匹配率: {tech_metrics['unit_match']:.4f}\n")
            f.write(f"语义相似度: {results['semantic_similarity']:.4f}\n")
            f.write(f"BERT F1分数: {results['bert_f1']}\n")
            f.write(f"ROUGE分数: {results['rouge']}\n")
            f.write(f"BLEU分数: {results['bleu']}\n")
            f.write('\n' + '=' * 50 + '\n\n')

    # 创建DataFrame
    df = pd.DataFrame(results_dict)
    
    # 设置Model列为索引
    df.set_index('Model', inplace=True)
    
    # 重命名列为中文名称
    df.rename(columns=key_metrics, inplace=True)
    
    # 保存为Excel文件
    df.to_excel(excel_file)
    
    print(f"评估结果已保存到: {excel_file}")
    print("\n评估结果概览:")
    print(df)

    # 选择要展示的关键指标
    key_indicators = [
        'term_coverage',          # 技术术语覆盖率
        'weighted_term_score',    # 技术术语权重得分
        'semantic_similarity',    # 语义相似度
        'bert_f1',               # BERT F1
        'rouge-l',               # ROUGE-L
        'bleu-4'                 # BLEU-4
    ]
    
    # 获取中文指标名称
    indicators_zh = [key_metrics[k] for k in key_indicators]
    
    # 准备数据并使用不同方法进行归一化
    plot_data = df[indicators_zh].copy()
    normalization_methods = ['minmax', 'zscore', 'robust', 'maxabs', 'log']
    
    # 创建一个字典来存储不同标准化方法的结果
    normalized_results = {}
    for method in normalization_methods:
        normalized_data = normalize_metrics(plot_data, indicators_zh, method=method)
        normalized_results[method] = normalized_data
        
        # 保存每种方法的结果
        output_file = f'eval_demo/normalized_evaluation_results_{method}.xlsx'
        normalized_data.to_excel(output_file)
        print(f"\n{method}标准化后的评估结果已保存到: {output_file}")
        print(f"\n{method}标准化后的评估结果概览:")
        print(normalized_data)
        
        # 绘制雷达图
        models = normalized_data.index
        values = normalized_data.values
        
        # 设置雷达图的角度
        angles = np.linspace(0, 2*np.pi, len(indicators_zh), endpoint=False)
        
        # 闭合雷达图
        values = np.concatenate((values, values[:, [0]]), axis=1)
        angles = np.concatenate((angles, [angles[0]]))
        
        # 创建图形
        plt.figure(figsize=(7, 5))
        ax = plt.subplot(111, projection='polar')
        
        # 绘制每个模型的雷达图
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, model in enumerate(models):
            ax.plot(angles, values[i], 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values[i], alpha=0.25, color=colors[i])
        
        # 设置雷达图的刻度和标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(indicators_zh, fontsize=10, rotation=45, position=(0, -0.15))
        
        # 设置刻度范围为0-1
        ax.set_ylim(0, 1)
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(0, 0.15))
        
        # 设置标题
        plt.title(f'模型评估关键指标对比（{method}标准化）', y=1.08)
        
        # 保存图形
        radar_file = f'eval_demo/radar_plot_{method}.png'
        plt.savefig(radar_file, dpi=300)
        print(f"\n雷达图已保存到: {radar_file}")
        
        # 关闭当前图形
        plt.close()
    
    print("\n所有标准化方法的结果已保存完成。")    