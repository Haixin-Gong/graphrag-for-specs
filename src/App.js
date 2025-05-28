import React, { useState } from 'react';
import { Layout, Upload, Button, Input, Select, message, Card, Typography } from 'antd';
import { UploadOutlined, SearchOutlined, BuildOutlined } from '@ant-design/icons';
import axios from 'axios';
import './App.css';

const { Content } = Layout;
const { TextArea } = Input;
const { Title } = Typography;

// 设置后端API的基础URL
const API_BASE_URL = 'http://192.168.1.121:5000';  // 使用实际的IP地址

function App() {
  const [fileList, setFileList] = useState([]);
  const [indexing, setIndexing] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [queryMethod, setQueryMethod] = useState('global');
  const [queryText, setQueryText] = useState('');
  const [queryResult, setQueryResult] = useState('');

  // 处理文件上传
  const handleFileUpload = async ({ file, fileList }) => {
    setFileList(fileList);
    
    if (file.status === 'done') {
      message.success(`${file.name} 上传成功`);
    } else if (file.status === 'error') {
      message.error(`${file.name} 上传失败`);
    }
  };

  // 自定义上传方法
  const customUpload = async ({ file, onSuccess, onError }) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onSuccess(response, file);
    } catch (error) {
      onError(error);
    }
  };

  // 处理索引创建
  const handleIndex = async () => {
    if (fileList.length === 0) {
      message.warning('请先上传文件');
      return;
    }

    setIndexing(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/index`, {
        files: fileList.map(file => file.name)
      });
      
      if (response.data.success) {
        message.success('索引创建成功！');
      } else {
        message.error('索引创建失败：' + response.data.message);
      }
    } catch (error) {
      message.error('索引创建失败：' + (error.response?.data?.message || error.message));
    }
    setIndexing(false);
  };

  // 处理查询
  const handleQuery = async () => {
    if (!queryText) {
      message.warning('请输入查询内容');
      return;
    }
    setQuerying(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/query`, {
        method: queryMethod,
        query: queryText
      });
      
      if (response.data.success) {
        setQueryResult(response.data.result);
      } else {
        message.error('查询失败：' + response.data.message);
      }
    } catch (error) {
      message.error('查询失败：' + (error.response?.data?.message || error.message));
      setQueryResult('');
    }
    setQuerying(false);
  };

  return (
    <Layout className="layout">
      <Content style={{ padding: '50px' }}>
        <Title level={2}>GraphRAG 知识库系统</Title>
        
        <Card title="文件上传" style={{ marginBottom: 20 }}>
          <Upload
            multiple
            fileList={fileList}
            onChange={handleFileUpload}
            customRequest={customUpload}
            accept=".txt"
          >
            <Button icon={<UploadOutlined />}>选择文件</Button>
          </Upload>
          <Button
            type="primary"
            onClick={handleIndex}
            loading={indexing}
            style={{ marginTop: 16 }}
            icon={<BuildOutlined />}
            disabled={fileList.length === 0}
          >
            创建索引
          </Button>
        </Card>

        <Card title="知识库查询">
          <Select
            value={queryMethod}
            onChange={setQueryMethod}
            style={{ width: 120, marginBottom: 16 }}
            options={[
              { value: 'global', label: 'Global模式' },
              { value: 'local', label: 'Local模式' }
            ]}
          />
          <TextArea
            placeholder="请输入您的问题..."
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            style={{ marginBottom: 16 }}
            rows={4}
          />
          <Button
            type="primary"
            onClick={handleQuery}
            loading={querying}
            icon={<SearchOutlined />}
          >
            查询
          </Button>
          
          {queryResult && (
            <div style={{ marginTop: 16 }}>
              <Title level={4}>查询结果：</Title>
              <div style={{ whiteSpace: 'pre-wrap' }}>{queryResult}</div>
            </div>
          )}
        </Card>
      </Content>
    </Layout>
  );
}

export default App; 