# PRAM 主预测代码

本项目将 `Test_20epoh_Kmeans.ipynb` 的主预测流程整理成可安装、可上传 GitHub 的 Python 包，**不包含消融实验、参数扫描和硬编码结果绘图**。完整命令及数据字段见 [English README](README.md)。

## 保留的流程

数据清洗与分箱 → 按年份划分训练/测试 → 验证集选择逻辑回归的 C → 筛选高风险拒绝样本 → K-means 选择靠近中心的真实样本 → Sparsemax 注意力记忆网络 → 保存模型并预测。

主网络保留 64 维表示、1 个注意力头和固定 `g=0.3`。拒绝样本不被当作已知标签参与监督训练。训练使用 BCE、Adam、验证损失早停；输出 AUC、KS、accuracy、precision、recall、F1，以及多随机种子的均值和标准差。保留原主代码中的 LR 参考指标，没有添加其他 benchmark。

## 安装与快速运行

在本目录打开终端，使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pram make-demo --output data/demo
pram train --config configs/demo.json \
  --accepted data/demo/accepted.csv --rejected data/demo/rejected.csv \
  --output runs/demo --device cpu
pram predict --checkpoint runs/demo/seed_42/model.pt \
  --input data/demo/applicants.csv --output runs/demo/predictions.csv
```

演示数据是合成数据，只用于检查程序运行，不能用来报告论文效果。Windows 用 `.venv\Scripts\activate` 激活虚拟环境。

## 使用真实数据

```bash
pram train --config configs/2014-2015.json \
  --accepted /path/to/accepted_2007_to_2018Q4.csv.gz \
  --rejected /path/to/rejected_2007_to_2018Q4.csv.gz \
  --output runs/2014-2015 --device auto
```

提供 `2012-2013.json`、`2013-2014.json`、`2014-2015.json` 三组配置，分别保留 notebook 的 K=8/12/13 和各组 5 个随机种子。共同设置为 top-k=3、20 epochs、batch size=10000、学习率=0.001、weight decay=0.0001、patience=3。

**三组配置中的 threshold=0.2 是可运行示例，不代表已确认的论文最终最优值。** 原 notebook 对多个阈值循环运行；这里去掉扫描，只运行配置指定的一个阈值。请在 JSON 中填入你要使用的值。

## 预处理与复现

原 notebook 有两个会影响结果的行为，项目没有悄悄替换：

1. 每年分别用当年已接受和拒绝样本拟合分箱，包括测试年份的特征分布。
2. 工作年限映射区分大小写，`1 YEAR` 与 `< 1 YEAR` 可以映射，但常见的 `1 year`、`< 1 year` 会被丢弃。

历史配置使用 `notebook_yearwise`，保留这些行为；它使用了测试年的特征信息，不等同于完全独立的未来数据评估。增加 `--preprocessing train_only` 则只在训练子集的已接受样本上拟合分箱，并规范工作年限字符串，适合新申请预测。两种模式的实验结果不能混为一谈。详见 [复现说明](docs/REPRODUCIBILITY.md)。

输出目录包含配置、依赖版本、各随机种子指标、训练日志及可加载模型。指标范围为 0–1，标准差使用 `ddof=0`。不会根据测试集选出一个“最佳种子”。

## 加载模型预测

输入 CSV 需要四个原始数值列：`fico_score`、`dti`、`loan_amnt`、`emp_length`。例如 DTI 为 15% 时填 `15`；工作年限填数值。不要提前取对数或分箱。附加的 ID 列会原样保留。

```bash
pram predict --checkpoint runs/2014-2015/seed_4507/model.pt \
  --binning train --input applicants.csv --output predictions.csv
```

`notebook_yearwise` 模型必须明确指定 `--binning train`（新申请使用训练年的分箱）或 `--binning test`（重现原测试年份流程）。`train_only` 模型可省略此参数。预测时不重新拟合分箱，缺失值会明确报错。

## 上传 GitHub

上传本目录的源码即可，原始数据、模型和运行结果已加入 `.gitignore`。没有上传到远程仓库，也没有替作者指定开源许可证。正式发布前补充你选择的许可证和核实后的论文引用信息。

