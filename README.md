# Telco Customer Churn Prediction

一个端到端客户流失预测项目，使用 IBM 公开的
Telco Customer Churn 数据集完成数据清洗、特征工程、模型比较、阈值预测、
模型持久化和 FastAPI（法斯特诶皮艾；接口服务框架）部署。

## 项目结果

| 指标 | 结果 |
| --- | ---: |
| 数据规模 | 7,043 条客户记录 |
| 特征数量 | 19 |
| 流失客户占比 | 26.54% |
| 最佳模型 | Logistic Regression |
| Accuracy | 73.74% |
| Precision | 50.34% |
| Recall | 78.34% |
| F1 | 61.30% |
| ROC-AUC | 84.14% |

项目比较了 Logistic Regression 和 Random Forest。两个模型都使用五折分层交叉验证，
最终选择交叉验证 ROC-AUC 更高的 Logistic Regression，并使用类别权重缓解正负样本不平衡。

## 完整流程

```mermaid
flowchart LR
    D[公开客户数据] --> C[数据清洗]
    C --> F[特征工程]
    F --> M[模型比较]
    M --> E[测试集评估]
    E --> S[保存模型]
    S --> A[FastAPI 服务]
    A --> P[流失概率与风险等级]
```

## 覆盖内容

- 缺失值处理。
- 数值特征标准化。
- 类别特征独热编码。
- 五折分层交叉验证。
- 逻辑回归与随机森林模型比较。
- Accuracy、Precision、Recall、F1 和 ROC-AUC 评估。
- 混淆矩阵和 ROC 曲线。
- 模型序列化与加载。
- FastAPI 健康检查、模型信息和预测接口。
- Pydantic 输入校验和异常响应。

## 运行

训练模型：

```powershell
.\run_train.ps1
```

运行测试：

```powershell
.\run_tests.ps1
```

启动预测服务：

```powershell
.\run_api.ps1
```

服务启动后访问：

- 接口文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`
- 模型指标：`http://127.0.0.1:8000/model-info`
- 预测接口：`POST http://127.0.0.1:8000/predict`

## 预测示例

```json
{
  "gender": "Female",
  "SeniorCitizen": "0",
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0,
  "TotalCharges": 180.0
}
```

响应包含：

```json
{
  "churn_probability": 0.0,
  "predicted_churn": false,
  "risk_level": "low",
  "model_version": "训练时间"
}
```

## 数据来源

数据集来自 IBM 公开项目：

`https://github.com/IBM/telco-customer-churn-on-icp4d`

数据只用于学习和项目展示，不包含真实生产客户数据。

## 目录结构

```text
telco-churn-ml-service/
├── churn/
│   ├── data.py                # 数据读取与特征定义
│   ├── train.py               # 训练、评估和模型保存
│   ├── api.py                 # FastAPI 服务
│   └── schemas.py             # 请求与响应校验
├── data/telco_churn.csv       # 公开数据
├── models/
│   ├── churn_pipeline.joblib  # 训练后的完整流水线
│   └── metrics.json           # 评估指标
├── docs/
│   ├── index.html             # 项目展示页
│   ├── model-evaluation.png   # ROC 曲线
│   └── test-report.html       # 自动化测试报告
└── tests/                     # 数据、模型和接口测试
```


## 项目边界

这是基于公开数据的个人机器学习项目，不代表企业生产模型。实际业务部署还需要考虑
数据漂移、模型监控、隐私合规和概率校准。
