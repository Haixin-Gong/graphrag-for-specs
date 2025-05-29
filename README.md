# GraphRAG Web Interface
![主界面](images/api.png)
这是一个基于GraphRAG的知识库系统的Web界面，支持文档上传、索引创建和知识查询功能。

## 功能特点

- 实现GraphRAG对Deepseek和GLM模型的支持
- 支持多文件上传（.txt格式）
- 支持创建知识库索引
- 提供两种查询模式：Global和Local
- 实时查询响应
- 行业术语词典的创建以及生成质量的多维度评估

## 系统要求

### 后端要求
- Python 3.6+
- Flask
- Flask-CORS

### 前端要求
- Node.js 14+
- npm 或 yarn

## 项目结构
```
graphrag/
├── frontend/              # 前端React应用
│   ├── src/              # 源代码
│   │   ├── index.js      # 应用入口点
│   │   ├── App.js        # 主应用组件
│   │   └── App.css       # 样式文件
│   ├── public/           # 静态文件
│   │   └── index.html    # HTML模板
│   └── package.json      # 前端依赖配置
├── server.py             # 后端Flask服务器
├── requirements.txt      # Python依赖
├── evaluation_metrics.py # 评估指标计算脚本
├── eval_demo/           # 评估结果目录
│   ├── evaluation_metrics.txt      # 评估详细结果
│   ├── evaluation_results.xlsx     # 原始评估数据
│   ├── normalized_evaluation_results_maxabs.xlsx  # 标准化后的评估数据
│   └── radar_plot_maxabs.png      # 评估雷达图
└── ragtest/              # GraphRAG工作目录
    └── input/           # 上传文件存储目录
```

## 安装步骤

1. 克隆仓库，添加 GraphRAG 对 Deepseek 和 GLM-embedding 模型的接口：
```bash
git clone https://github.com/Haixin-Gong/graph-rag.git
cd graphrag
```

2. 安装后端依赖：
```bash
pip install -r requirements.txt
```

3. 安装前端依赖：
```bash
cd frontend
npm install
```

## 扩展 GraphRAG 的模型接口
初始化 GraphRAG
```bash
graphrag init --root ./ragtest
```
- 将 Deepseek 和 GLM 的 API KEY，model_base, model 传入 settings.yaml 和 .envs (可选择)
- 修改 GraphRAG 的 *tiktoken* 工具中的 *model.py* 实现，添加以下信息：
```bash
# Add deepseek-chat model
"deepseek-chat": "cl100k_base",
"embedding-3": "cl100k_base"
```

## 运行应用

1. 启动后端服务器：
```bash
# 在项目根目录下
python server.py
```
后端服务器将在 http://localhost:5000 运行

2. 启动前端开发服务器：
```bash
# 在frontend目录下
export NODE_OPTIONS=--openssl-legacy-provider
npm start
```
前端应用将在 http://localhost:3000 运行

## 使用指南

### 文件上传
1. 点击"选择文件"按钮
2. 选择一个或多个.txt文件
3. 文件会自动上传到服务器

### 创建索引
1. 确保已经上传了文件
2. 点击"创建索引"按钮
3. 等待索引创建完成

### 知识查询
1. 选择查询模式：
   - Global模式：在整个文档范围内查询
   - Local模式：在相关上下文范围内查询
2. 在输入框中输入您的问题
3. 点击"查询"按钮获取答案

## API文档

### 后端API端点

#### 1. 文件上传
- 端点：`POST /upload`
- 请求：multipart/form-data
- 响应：
```json
{
    "success": true/false,
    "message": "上传状态信息"
}
```

#### 2. 创建索引
- 端点：`POST /index`
- 响应：
```json
{
    "success": true/false,
    "message": "索引创建状态"
}
```

#### 3. 知识查询
- 端点：`POST /query`
- 请求体：
```json
{
    "method": "global/local",
    "query": "查询问题"
}
```
- 响应：
```json
{
    "success": true/false,
    "result": "查询结果",
    "message": "错误信息（如果有）"
}
```

## 常见问题解决

1. 前端启动错误
   - 问题：OpenSSL相关错误
   - 解决：使用 `export NODE_OPTIONS=--openssl-legacy-provider`

2. 后端连接问题
   - 检查后端服务是否运行
   - 确认API地址配置正确
   - 检查CORS设置

3. 文件上传失败
   - 确保文件格式为.txt
   - 检查文件大小是否超限
   - 确保上传目录有写入权限

## 系统评估

系统采用多维度评估方法，包括：
- 技术术语覆盖率：评估生成内容对关键技术术语的覆盖程度
- 技术术语权重得分：考虑术语在文档中的重要性
- 语义相似度：使用BERT模型计算语义相似度
- BERT F1分数：评估生成内容的准确性
- ROUGE-L分数：评估生成内容的连贯性
- BLEU-4分数：评估生成内容的流畅度

为了使不同评估指标具有可比性，采用最大绝对值标准化（MaxAbs）方法：

```math
X_{normalized} = \frac{X}{max(|X|)}
```

其中X为原始指标值，max(|X|)为所有指标值绝对值中的最大值。

### 评估结果
<img src="images/radar_plot_maxabs.png">

## 许可证

Following MIT License

## 更新日志

### v1.0.0
- 初始版本发布
- 基本功能实现：文件上传、索引创建、知识查询