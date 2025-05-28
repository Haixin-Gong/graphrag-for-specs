from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__)
CORS(app)  # 启用CORS以允许前端访问

# 配置文件上传目录
UPLOAD_FOLDER = './ragtest/input'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 根路由，用于测试API是否正常工作
@app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'message': 'GraphRAG API is running'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有文件被上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'}), 400
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify({'success': True, 'message': '文件上传成功'})

@app.route('/index', methods=['GET', 'POST'])
def create_index():
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Use POST method to create index'
        })

    try:
        # 运行GraphRAG的index命令
        result = subprocess.run(['graphrag', 'index', '--root', './ragtest'], 
                             capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': '索引创建成功'})
        else:
            return jsonify({'success': False, 'message': f'索引创建失败: {result.stderr}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Use POST method to query'
        })

    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': '没有提供查询参数'}), 400

        query_text = data.get('query')
        if not query_text:
            return jsonify({'success': False, 'message': '查询内容不能为空'}), 400

        method = data.get('method', 'global')
        
        # 运行GraphRAG的query命令
        result = subprocess.run([
            'graphrag', 'query',
            '--root', './ragtest',
            '--method', method,
            '--query', query_text
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # 处理输出结果
            output = result.stdout
            
            # 移除不需要的内容
            cleaned_lines = []
            is_vector_store_info = False  # 用于标记是否在vector store信息块中
            
            for line in output.split('\n'):
                # 跳过空行
                if not line.strip():
                    continue
                
                # 跳过vector store配置信息
                if '"default_vector_store"' in line or line.strip().startswith('{') or line.strip().startswith('}'):
                    is_vector_store_info = True
                    continue
                
                if is_vector_store_info:
                    if line.strip().endswith('}'):
                        is_vector_store_info = False
                    continue
                
                # 跳过特定前缀的行
                if any(line.strip().startswith(prefix) for prefix in [
                    'INFO:', 'SUCCESS:', '"type":', '"db_url":', '"url":', 
                    '"audience":', '"container_name":', '"database_name":', '"overwrite":'
                ]):
                    continue
                
                cleaned_lines.append(line)
            
            # 重新组合处理后的文本，使用两个换行符分隔段落
            cleaned_output = '\n\n'.join(cleaned_lines)
            
            return jsonify({'success': True, 'result': cleaned_output})
        else:
            return jsonify({'success': False, 'message': f'查询失败: {result.stderr}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 处理404错误
@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': '接口不存在'}), 404

# 处理405错误
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'success': False, 'message': '请求方法不允许'}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)