# 数据说明

## 来源
- 数据集：`online_shopping_10_cats`（中文电商评论，10 个品类，字段 `cat, label, review`；`label` 1=正向，0=负向）
- 仓库：[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) → `datasets/online_shopping_10_cats/online_shopping_10_cats.zip`
- 许可：该仓库未附带明确的开源许可证文件（截至 2026-11-10 通过 GitHub API 查询 license 字段为空）。本项目仅将其用于学习与作品集演示，**不在本仓库中再分发完整数据**；如需商用请自行联系原作者确认授权。

## 仓库里有什么
| 文件 | 是否提交 | 说明 |
|---|---|---|
| `raw/online_shopping_10_cats.zip` | ❌（.gitignore） | 原始数据，运行脚本自动下载 |
| `processed/shampoo_clean.csv` | ❌（.gitignore） | 清洗后的洗发水评论（9,984 条） |
| `eval/eval_set.csv` | ✅ | 固定评测集：300 条（正/负各 150，random_state=42） |
| `eval/tag_labeling_sheet.csv` | ✅ | 50 条待人工标注问题标签的表格（从评测集抽取） |

## 如何下载完整数据（Windows）
```powershell
.venv\Scripts\python scripts\prepare_data.py
```
脚本会：下载 zip 到 `data/raw/` → 只保留 `cat == "洗发水"` → 去空、去重、去近似重复 → 生成评测集与标注表 → 把每一步的行数写入 `outputs/data_quality.md`。
如果下载失败，脚本会直接退出并报错，不会生成任何假数据；也可以手动下载 zip 放到 `data/raw/`。

## 数据质量表
见 [`outputs/data_quality.md`](../outputs/data_quality.md)。

## 标签标注说明（tag_labeling_sheet.csv）
用 Excel 打开，按每条评论填写 `human_tag_1`（必填）和 `human_tag_2`（可选），只能从以下 13 个标签中选：
`去屑 控油 香味 质地 起泡 洗后发质 头皮刺激/过敏 包装 物流 价格 真伪 客服 其他`
填完保存为 CSV（UTF-8），运行：
```powershell
.venv\Scripts\python scripts\tag_agreement.py --version v2
```
