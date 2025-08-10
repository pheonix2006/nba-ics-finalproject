# NBA数据分析项目 - Team 9

## 项目简介

本项目是一个综合性的NBA数据分析项目，包含以下几个主要部分：
- NBA球员身体数据分析
- NBA球员和球队表现数据分析  
- 机器学习模型预测分析
- Transformer模型应用

## 环境要求

### Python版本
- Python 3.8 或更高版本

### 必需的Python库

#### 数据处理和分析
```bash
pip install pandas
pip install numpy
```

#### 机器学习
```bash
pip install scikit-learn
```

#### 深度学习 (PyTorch)
```bash
# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或者GPU版本 (如果有CUDA支持)
pip install torch torchvision torchaudio
```

#### 数据可视化
```bash
pip install matplotlib
pip install seaborn
```

#### Jupyter Notebook支持
```bash
pip install jupyter
pip install ipykernel
```

### 一键安装所有依赖

创建并激活虚拟环境（推荐）：
```bash
# 创建虚拟环境
python -m venv nba_analysis_env

# 激活虚拟环境 (Windows)
nba_analysis_env\Scripts\activate

# 激活虚拟环境 (macOS/Linux)
source nba_analysis_env/bin/activate
```

安装所有必需的库：
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
## 使用说明

### 1. 环境配置
按照上述环境要求安装所有必需的库。

### 2. 启动Jupyter Notebook
```bash
jupyter notebook
```

### 3. 运行分析
- **身体数据分析**: 打开 `nba final project by team 9/body data analysis/Part B-2 无注释.ipynb`
- **NBA数据分析**: 打开 `nba final project by team 9/nba analysis/analysis.ipynb`
- **机器学习分析**: 打开 `nba final project by team 9/nba analysis/ml analysis.ipynb`
- **Transformer模型**: 打开 `lib resources/Lab11 (1).ipynb`

## 主要功能

### 数据处理
- CSV文件读取和处理
- 数据清洗和预处理
- 球队名称标准化
- 数据合并和排序

### 数据分析
- 球员身体数据统计分析
- 球员表现数据分析
- 球队数据分析
- 数据可视化

### 机器学习
- 随机森林回归模型
- 特征工程和数据标准化
- 模型评估和预测

### 深度学习
- Transformer模型实现
- PyTorch框架应用
- 神经网络训练和测试

## 注意事项

1. 确保所有CSV数据文件都在正确的路径下
2. 运行notebook前请确认当前工作目录
3. 如果遇到内存不足问题，可以考虑分批处理数据
4. GPU版本的PyTorch需要相应的CUDA驱动支持

## 故障排除

### 常见问题
1. **ModuleNotFoundError**: 确保已安装所有必需的库
2. **文件路径错误**: 检查CSV文件是否在正确位置
3. **内存错误**: 尝试减少数据集大小或使用更强大的硬件

### 获取帮助
如果遇到问题，请检查：
1. Python版本是否符合要求
2. 所有依赖库是否正确安装
3. 数据文件是否完整

## 团队信息
- 项目团队：Team 9
- 项目类型：NBA数据分析最终项目