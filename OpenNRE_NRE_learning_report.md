# 基于深度学习的 NRE 模型学习报告：OpenNRE

课程作业：第二次作业  
项目仓库：https://github.com/thunlp/OpenNRE/  
分析对象：OpenNRE 中的神经关系抽取模型、代码结构、模型结构与实际调用设计

## 1. 任务背景

NRE 是 Neural Relation Extraction，即神经关系抽取。它的目标是从自然语言文本中抽取实体之间的语义关系，并形成结构化三元组：

```text
实体1 - 关系 - 实体2
```

例如句子：

```text
Bill Gates founded Microsoft.
```

如果已知实体 `Bill Gates` 和 `Microsoft` 的位置，关系抽取模型需要判断二者之间的关系是：

```text
Bill Gates - founder_of - Microsoft
```

关系抽取是知识图谱构建中的基础环节。文本中大量事实本来是非结构化的，NRE 模型可以把这些文本事实转成可查询、可推理、可存储的关系数据。

OpenNRE 是清华 THUNLP 开源的神经关系抽取工具包。它支持两类典型设置：

- 句子级关系抽取：给定一个句子和两个实体，预测二者关系；
- 包级关系抽取：远程监督场景下，把同一实体对对应的多个句子组成一个 bag，再判断实体对关系。

## 2. OpenNRE 代码结构

OpenNRE 的核心代码在 `opennre/` 下，结构比较清晰：

```text
OpenNRE/
├── example/                 # 训练和测试示例脚本
├── benchmark/               # 数据集下载脚本
├── pretrain/                # 预训练模型/词向量下载脚本
├── tests/                   # 简单推理测试
├── opennre/
│   ├── encoder/             # 文本编码器：CNN、PCNN、BERT
│   ├── model/               # 关系抽取模型：Softmax、BagAttention 等
│   ├── framework/           # 数据加载、训练、评估流程
│   ├── module/              # 底层神经网络模块：CNN、RNN、pooling
│   ├── tokenization/        # 分词与 token-id 转换
│   ├── pretrain.py          # 预训练模型/数据下载与 get_model 接口
│   └── __init__.py
├── requirements.txt
└── setup.py
```

我理解它的设计思路是：把“文本编码”“关系分类”“训练框架”拆开。这样换模型时可以只换其中一层。例如 CNN encoder 可以接 SoftmaxNN，也可以接 BagAttention；BERT encoder 也可以接同样的分类层。

## 3. 数据格式

OpenNRE 的输入样本通常包含：

```python
{
    "text": "Bill Gates founded Microsoft.",
    "h": {"pos": (0, 10)},
    "t": {"pos": (19, 28)},
    "relation": "founder_of"
}
```

其中：

- `text` 是原句；
- `h` 是 head entity；
- `t` 是 tail entity；
- `pos` 是实体在原文中的字符位置或 token 位置；
- `relation` 是训练时的标签。

推理时不需要 `relation` 字段，只需要文本和两个实体的位置。

在远程监督 bag-level 数据中，多个句子可能共享同一实体对。例如多个新闻句子都提到 `Apple` 和 `Tim Cook`，系统会把这些句子合成一个 bag。模型不再只判断单句关系，而是判断整个实体对关系。

## 4. 模型结构

OpenNRE 的模型可以拆成三层：

```text
输入文本 + 实体位置
        ↓
Encoder：得到句子表示
        ↓
Relation Model：得到关系类别分数
        ↓
输出关系标签 + 置信度
```

### 4.1 CNNEncoder

代码位置：`opennre/encoder/cnn_encoder.py`

CNNEncoder 使用三类 embedding：

- word embedding：词向量；
- pos1 embedding：每个词相对 head entity 的位置；
- pos2 embedding：每个词相对 tail entity 的位置。

前向过程：

```text
word embedding + position embedding
        ↓
1D Convolution
        ↓
ReLU
        ↓
Max Pooling
        ↓
句子向量
```

它的核心代码逻辑是：

```python
x = torch.cat([word_embedding, pos1_embedding, pos2_embedding], 2)
x = x.transpose(1, 2)
x = relu(conv1d(x))
x = max_pool(x)
```

CNNEncoder 的优点是结构简单、推理速度快。缺点是上下文建模能力弱于 BERT。

### 4.2 PCNNEncoder

代码位置：`opennre/encoder/pcnn_encoder.py`

PCNN 是 Piecewise CNN。它和普通 CNN 的区别在于 pooling 方式。

普通 CNN 对整句做一次 max pooling。PCNN 会根据两个实体的位置，把句子划成三段：

```text
实体1之前 / 两个实体之间 / 实体2之后
```

然后分别做 max pooling，最后拼接三段表示。

这样做的原因是：关系抽取高度依赖实体之间及其上下文区域。分段池化可以保留更多实体相对位置信息。

PCNN 的输出维度是普通 CNN hidden size 的 3 倍，因为它拼接了三段池化结果。

### 4.3 BERTEncoder 与 BERTEntityEncoder

代码位置：`opennre/encoder/bert_encoder.py`

BERTEncoder 使用预训练 BERT 生成上下文表示。OpenNRE 提供两种池化策略：

- `BERTEncoder`：使用 BERT 的 pooled output，类似 `[CLS]` 表示；
- `BERTEntityEncoder`：取 head entity 和 tail entity 起始位置的 hidden state，然后拼接。

BERTEntityEncoder 更适合关系抽取，因为它显式使用实体位置表示：

```text
head entity hidden state + tail entity hidden state
        ↓
Linear
        ↓
sentence representation
```

实体位置通过特殊 token 标记：

```text
[unused0] head [unused1] ... [unused2] tail [unused3]
```

这种做法让 BERT 明确知道当前要判断的是哪两个实体，而不是只理解整句话。

## 5. 关系分类模型

### 5.1 SoftmaxNN：句子级关系分类

代码位置：`opennre/model/softmax_nn.py`

SoftmaxNN 是句子级关系抽取模型。它接收 encoder 输出的句子向量，然后用线性层映射到关系类别：

```text
sentence representation
        ↓
Dropout
        ↓
Linear(hidden_size, num_class)
        ↓
Softmax
        ↓
关系类别
```

核心代码逻辑：

```python
rep = self.sentence_encoder(*args)
rep = self.drop(rep)
logits = self.fc(rep)
```

推理时调用：

```python
model.infer(item)
```

返回：

```text
(relation_name, confidence_score)
```

适合场景：输入是一句话，实体对已经确定，需要判断这句话表达了什么关系。

### 5.2 BagAttention：包级关系抽取

代码位置：`opennre/model/bag_attention.py`

BagAttention 用在远程监督关系抽取中。远程监督数据噪声较大，因为知识库中存在实体关系，不代表每个包含该实体对的句子都表达了这个关系。

例如：

```text
Apple CEO Tim Cook spoke at the event.
Tim Cook visited an Apple store.
Apple announced a new product.
```

这些句子都可能和 `Apple - Tim Cook` 相关，但并不是每个句子都表达 `CEO_of`。

BagAttention 的做法是：对同一实体对的多个句子分别编码，然后使用 attention 给不同句子分配权重，重点关注更能表达目标关系的句子。

训练时，attention 会根据当前关系 label 选择相关句子：

```text
bag 内多个句子表示
        ↓
relation-specific attention
        ↓
bag representation
        ↓
relation classifier
```

这就是 OpenNRE README 里强调的 ATT 方法。

## 6. 训练框架

OpenNRE 的训练框架在 `opennre/framework/` 下。

### 6.1 SentenceRE

代码位置：`opennre/framework/sentence_re.py`

SentenceRE 负责句子级关系抽取训练：

```text
SentenceRELoader
        ↓
model forward
        ↓
CrossEntropyLoss
        ↓
SGD / Adam / AdamW
        ↓
validation
        ↓
保存最佳 checkpoint
```

评估指标包括：

- accuracy；
- micro precision；
- micro recall；
- micro F1。

### 6.2 BagRE

代码位置：`opennre/framework/bag_re.py`

BagRE 负责包级关系抽取训练。它使用 `BagRELoader` 把样本组织成 bag。

评估时输出：

- AUC；
- max micro F1；
- max macro F1；
- P@100；
- P@200；
- P@300。

包级关系抽取更接近知识图谱自动构建，因此它更重视排序质量和 precision-recall 曲线。

## 7. 训练入口脚本

OpenNRE 的训练脚本在 `example/` 目录。

### 7.1 句子级 BERT 模型训练

脚本：

```text
example/train_supervised_bert.py
```

典型命令：

```bash
python example/train_supervised_bert.py \
    --pretrain_path bert-base-uncased \
    --dataset wiki80
```

这会构建：

```text
BERTEntityEncoder / BERTEncoder
        +
SoftmaxNN
        +
SentenceRE framework
```

适合标准监督关系抽取任务。

### 7.2 包级 CNN/PCNN-ATT 模型训练

脚本：

```text
example/train_bag_cnn.py
```

典型命令：

```bash
python example/train_bag_cnn.py \
    --metric auc \
    --dataset nyt10m \
    --batch_size 160 \
    --lr 0.1 \
    --weight_decay 1e-5 \
    --max_epoch 100 \
    --max_length 128 \
    --seed 42 \
    --encoder pcnn \
    --aggr att
```

这会构建：

```text
PCNNEncoder
        +
BagAttention
        +
BagRE framework
```

适合远程监督数据，例如 NYT10 / NYT10m。

## 8. 推理流程

OpenNRE 提供预训练模型加载接口：

```python
import opennre

model = opennre.get_model("wiki80_bertentity_softmax")
result = model.infer({
    "text": "Steve Jobs founded Apple in California.",
    "h": {"pos": (0, 10)},
    "t": {"pos": (19, 24)}
})

print(result)
```

输出形式：

```text
("founder", score)
```

推理必须提供实体位置。OpenNRE 本身主要做“给定实体对后的关系分类”，不是完整的信息抽取流水线。实际系统中通常还要接：

```text
文本清洗 -> 分句 -> NER 实体识别 -> 实体对生成 -> OpenNRE 关系分类 -> 三元组入库
```

## 9. 实际场景调用设计：企业新闻知识图谱构建

我设计的场景是：从财经新闻中自动抽取公司、人物、地点之间的关系，用于企业知识图谱和舆情分析。

### 9.1 场景目标

输入新闻：

```text
Tesla CEO Elon Musk met with officials in Shanghai to discuss the expansion of the company factory.
```

希望抽取：

```text
Elon Musk - employee_of / CEO_of - Tesla
Tesla - located_in / factory_in - Shanghai
```

这类结构化关系可以用于：

- 企业人物关系图谱；
- 公司地点布局分析；
- 新闻检索增强；
- 投资研究中的事件抽取；
- 问答系统，例如“某公司 CEO 是谁”。

### 9.2 系统流程

```text
新闻文本
  ↓
分句
  ↓
NER 识别实体：公司、人名、地点
  ↓
枚举实体对
  ↓
调用 OpenNRE 判断关系
  ↓
过滤低置信度结果
  ↓
输出三元组
  ↓
写入知识图谱或数据库
```

### 9.3 调用示例

示例脚本见同目录：

```text
opennre_business_news_demo.py
```

核心调用逻辑：

```python
model = opennre.get_model("wiki80_bertentity_softmax")

item = {
    "text": "Steve Jobs founded Apple in California.",
    "h": {"pos": (0, 10)},
    "t": {"pos": (19, 24)}
}

relation, score = model.infer(item)
```

实际部署时需要注意：

- OpenNRE 需要明确实体 span，因此前面要接 NER；
- 模型训练数据是英文 Wiki80/TACRED/NYT 等，直接处理中文新闻效果不会好；
- 如果目标是中文企业新闻，需要重新构造中文关系数据集并微调；
- 低置信度结果应过滤，例如 `score < 0.6` 不入库；
- 关系标签需要和业务 schema 对齐，例如 `founder_of`、`headquarters_in`、`subsidiary_of`。

## 10. 优点与不足

### 优点

- 模块化清楚，encoder、model、framework 分层合理；
- 同时支持 CNN/PCNN 和 BERT；
- 同时支持句子级和远程监督 bag-level；
- 示例脚本完整，便于复现实验；
- `get_model` 接口降低了预训练模型使用门槛。

### 不足

- 代码依赖较旧，例如 `torch==1.6.0`、`transformers==3.4.0`；
- 数据读取使用 `eval(line)`，安全性和鲁棒性不如 `json.loads`；
- 默认下载依赖 `wget`，在 Windows 环境下不方便；
- OpenNRE 只做关系分类，不负责 NER 和实体链接；
- 中文业务场景需要额外数据和训练，不适合直接拿英文模型硬套。

## 11. 总结

OpenNRE 是一个典型的深度学习关系抽取工具包。它的核心不是某一个单独模型，而是一套可组合框架：

```text
Encoder 提取句子/实体表示
Model 完成关系分类或 bag-level 聚合
Framework 负责训练、评估和 checkpoint
```

从模型角度看，CNN/PCNN 代表传统神经关系抽取方法，BERTEntityEncoder 代表预训练语言模型方法，BagAttention 则解决远程监督场景中的噪声句子选择问题。

如果用于实际业务，OpenNRE 更适合作为关系分类模块嵌入到完整信息抽取系统中。完整系统还需要 NER、实体链接、置信度过滤和知识图谱存储。对于课程作业，建议重点展示它的模块化代码结构、CNN/PCNN/BERT 模型结构，以及企业新闻知识图谱这一实际调用场景。

