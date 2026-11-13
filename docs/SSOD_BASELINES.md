# SSOD 基线调研笔记（2026-11-13）

背景：DsPCBSD+（9 类，约 8.2k 张 226x226 训练图，10% 标注 = 821 张）；YOLOv8n 学生 + RF-DETR(DINOv2) 离线伪标签 = 0.430 mAP50:95，纯监督 0.352，YOLOv8m 标注器 0.404。约束：Ultralytics、60 epoch、imgsz 320、Kaggle 单 T4、每次 1–2 h、Python 3.13 / torch 2.11。

标记约定：**[已核实]** = 本次直接从 GitHub API / 仓库原始文件 / 论文原文读到；**[摘要级]** = 只看到摘要或搜索摘要，未读正文表格；**[未核实]** = 推断或没查到；**[判断]** = 我的工程判断，没有实际安装运行验证。
注意：仓库日期用的是 GitHub 搜索 API 的 `pushed_at`（任意分支最后一次 push），不是逐条 commit 核对。本次没有在 Python 3.13 / torch 2.11 上实际安装任何仓库，所有"能否安装"的结论都是 [判断]。

---

## 1. 标准 SSOD 方法与代码可用性

| 方法 | 仓库 | pushed_at / 状态 [已核实] | 框架 / 官方环境 | 学生检测器 | 许可 | Py3.13 / torch2.11 [判断] |
|---|---|---|---|---|---|---|
| Unbiased Teacher v1 (ICLR'21) | https://github.com/facebookresearch/unbiased-teacher | 2021-09-10，**已归档** | Detectron2 | Faster R-CNN | MIT | 需源码编译 detectron2（PyPI 上无 detectron2 包，已核实 404）；老代码 + 新 torch 风险高 |
| Unbiased Teacher v2 (CVPR'22) | https://github.com/facebookresearch/unbiased-teacher-v2 | 2022-07-27，**已归档** | Detectron2 | FCOS / Faster R-CNN | MIT | 同上 |
| Soft Teacher (ICCV'21) | https://github.com/microsoft/SoftTeacher | 2024-06-09，**已归档** | MMDetection 2.x（原仓库）；MMDet 3.x 内置重实现 | Faster R-CNN | MIT | 很难：见下方 MMDet 说明 |
| Dense Teacher (ECCV'22) | https://github.com/Megvii-BaseDetection/DenseTeacher | 2022-12-08，未归档但停更 | cvpods（旷视自研） | FCOS | Apache-2.0 | 基本不可行 |
| Consistent-Teacher (CVPR'23) | https://github.com/Adamdad/ConsistentTeacher | 2023-07-19，停更 | MMDetection 2.x | RetinaNet 系 | Apache-2.0 | 很难 |
| PseCo (ECCV'22) | https://github.com/ligang-cs/PseCo | 2022-08-26，停更 | MMDetection 2.x | Faster R-CNN | Apache-2.0 | 很难 |
| **Efficient Teacher** (arXiv 2302.07577) | https://github.com/AlibabaResearch/efficientteacher | 2023-03-28；814 star；131 open issue；作者基本不再回复（最新 issue 2025-11-13 仍 open） | 自研，基于 YOLOv5 代码 + YACS；requirements 只写 `torch>=1.9.0`；官方验证环境是 py3.7 + torch1.11 的 docker | YOLOv5 n/s/m/l/x（靠 depth/width multiple）；README 称支持 YOLOX/v6/v7/v8，但 **SSOD 配置只有 YOLOv5l** | **GPL-3.0** | 可以装（纯 Python，无编译扩展），但要打补丁，见 §2 |
| Semi-DETR (CVPR'23) | https://github.com/JCZ404/Semi-DETR | 2024-11-26 | MMDetection 2.x + 自定义 CUDA 算子 | DINO-DETR | MIT | 很难，且 T4 上跑不动 |
| MixPL (arXiv 2312.07006) | https://github.com/Czm369/MixPL | 2024-03-07 | MMDetection 3.x | Faster R-CNN / FCOS / DINO 等 | Apache-2.0 | 很难 |
| LabelMatch (CVPR'22) | https://github.com/hikvision-research/SSOD | 2022-12-07 | MMDetection 2.x | Faster R-CNN | Apache-2.0 | 很难 |

MMDetection 系说明 [已核实日期，兼容性为判断]：`open-mmlab/mmdetection` 最后 push 2024-08-21，PyPI 最新 mmdet 3.3.0（2024-01-05 上传）；mmcv PyPI 最新 2.2.0（2024-04-24），PyPI 上**只有源码包**，要本地编译 CUDA 算子。mmcv 仓库 2026-09 仍有 push，但没有新发行版。在 Kaggle 的 Python 3.13 + torch 2.11 上编译 mmcv 并让 mmdet 3.3.0 通过版本断言，我估计要半天以上且可能失败；现实做法是另建 Python 3.10 + torch 2.1 的 conda/uv 环境——这在 Kaggle 上可行但要占用每次会话的安装时间。

**YOLOv8 / YOLO11 / RT-DETR 专用的 SSOD：没有找到可信、维护中的仓库。**
- Ultralytics 官方没有半监督训练功能 [未在 issue 中找到任何官方实现；搜索未命中]。
- GitHub 搜索 "semi-supervised yolov8 / ultralytics / mean teacher yolo" 得到的全是 0 star 的个人仓库 [已核实]：例如 `Connor1999che/Semi_supervised_yolov8`（MIT，2026-08，FixMatch/Mean-Teacher + GhostNetV2，要求 python 3.10，无论文背书）、`Avinash-0514/semi-supervised-varroa-detection`（README 自述"只是模板，不含原始代码"）、`munkhuu1126/yolov8-ssl`（作者自述失败）。都不能当作审稿人认可的基线。
- 最接近的"在 Ultralytics YOLO 上做 mean teacher"的已发表先例：蓝莓检测基准 arXiv 2509.20580，对 YOLOv8–v12 共 25 个模型用 "Unbiased Mean Teacher" 式 EMA 教师（置信度阈值 0.9，20 epoch 后降到 0.7），Ultralytics 8.0.196，mAP50 变化 **−1.4 到 +2.9**，25 个里 14 个提升 [已核实，来自论文 HTML]。仓库 https://github.com/rogermu789/BlueberryBenchmark 存在但 README 只有标题、无许可证 [已核实]，不能直接拿来用。
- SUD-YOLO（Applied Sciences 2025, https://www.mdpi.com/2076-3417/15/3/1065）：YOLOv8 + Mean Teacher，[摘要级]，未找到代码。
- RT-DETR 的半监督：上面蓝莓论文对 RT-DETR 用的是 Semi-DETR；没有找到 Ultralytics RT-DETR 的 SSOD 实现 [未核实是否存在]。
- 其他对比性综述：arXiv 2601.13380 "Practical Insights into SSOD Approaches" 比较 MixPL / Semi-DETR / Consistent-Teacher [摘要级]——说明这三个是 2026 年审稿人眼中的"现代"SSOD 代表，但都不是 YOLO。

工具类：
- Autodistill https://github.com/autodistill/autodistill ：Apache-2.0，pushed 2026-09-29，PyPI 最新 0.1.29（2024-11-26）。基础模型 Grounding DINO / GroundedSAM(2) / OWL-ViT / OWLv2 / DETIC 等，目标模型 YOLOv8 / YOLOv5 / YOLO-NAS / DETR。纯文本提示零样本标注，不用标注数据，README 无精度数字 [已核实]。
- lightly-train https://github.com/lightly-ai/lightly-train ：AGPL-3.0，2026-11 活跃，描述为 "YOLO, ViTs, RT-DETR, DINOv3 的预训练、微调、蒸馏" [只核实了仓库描述]。它做的是特征蒸馏预训练，不是伪标签，但属于"基础模型 -> 小检测器"的相邻工作，审稿人可能提到。
- RF-DETR https://github.com/roboflow/rf-detr ：Apache-2.0，活跃。

---

## 2. Efficient Teacher 细节

来源：仓库 README、`configs/`、`requirements.txt`、`train.py`，论文 PDF（arXiv 2302.07577）。

**能否用 YOLO 格式数据 + 有标/无标划分训练 YOLOv5n/s：能，但没有现成配置** [已核实]
- 数据：`Dataset.train` 指向有标注图像路径列表 txt，`Dataset.target` 指向无标注图像路径列表 txt（`find <dir> -name "*.jpg" >> unlabel.txt`），`val` 同理；标签为 YOLO txt 格式（沿用 YOLOv5 的 images/labels 目录约定）。改 `nc`、`names`。
- SSOD 配置只提供 YOLOv5l 的：`configs/ssod/custom/yolov5l_custom_ssod.yaml`、`yolov5l_transfer_ssod.yaml`、COCO 1/2/5/10%、VOC、Cityscapes。**没有 yolov5n/s 的 SSOD yaml**（我试取 `yolov5s_custom_ssod.yaml` 返回 404）。要自己把 `depth_multiple/width_multiple` 改成 0.33/0.25（n）或 0.33/0.50（s），这两个数值来自仓库的 `configs/sup/public/yolov5n_coco.yaml`。
- 需要一个 `SSOD:` 配置块，关键键：`train_domain: True`、`nms_conf_thres: 0.1`、`nms_iou_thres: 0.65`、`teacher_loss_weight: 1.0`、`cls/box/obj_loss_weight: 0.3/0.05/0.7`、`loss_type: 'ComputeStudentMatchLoss'`、`ignore_thres_low: 0.1`、`ignore_thres_high: 0.6`、`pseudo_label_with_obj/bbox: True`、`epoch_adaptor: True`、`ema_rate: 0.999`、`cosine_ema: True`、`ssod_hyp`（mosaic 1.0、cutout 0.5、autoaugment 0.5、scale 0.8）；`hyp.burn_epochs` 控制 burn-in。
- 启动：`python train.py --cfg <yaml>`（README 示例是 8 卡 DDP：`python -m torch.distributed.launch ...`；单卡直接跑 train.py，`LOCAL_RANK` 默认 -1）。验证：`python val.py --cfg ... --weights ... --val-ssod`。
- 预训练权重：release 1.0 提供 `efficient-yolov5n.pt`、`efficient-yolov5s.pt`（COCO 监督）等；用标准 YOLOv5 权重要先过 `scripts/mula_convertor/convert_pt_to_efficient.py`。
- YOLOv8：只有监督配置 `configs/sup/public/yolov8m_coco.yaml`（`ComputeTalLoss`）。YOLOv8 的 SSOD 没有配置；issue #11「支持yolov8半监督训练吗」、#70「yolov8半监督的效果如何」、#125「修改yolov8发现缺少loss文件」、#160「yolov8」都指向这一点 [只核实了 issue 标题，未读回复内容]。**结论：Efficient Teacher 下的学生只能是 YOLOv5n/s，不是 YOLOv8n。**

**训练长度** [已核实]
- 论文：8 张 V100，每次迭代有标 32 + 无标 32（1:1），lr 0.01，最多 300 epoch，EMA 0.999，从头训练。
- 仓库 COCO 10% 配置：`epochs: 300`、`burn_epochs: 0`、batch 32、img 640。自定义配置：`epochs: 20`、`burn_epochs: 10`、batch 128。迁移配置：`epochs: 30`、`burn_epochs: 10`、从 `efficient-yolov5l-obj365.pt` 起步。README 文字里又说"默认先 220 epoch burn-in"——README 与配置不一致。
- 对我们的预算：8.2k 张、imgsz 320、YOLOv5n/s、单 T4，每 epoch 同时过有标 + 无标 + 教师前向，[判断] 约为同规模监督训练的 2–3 倍耗时；60 epoch（如 20 burn-in + 40 SSOD）估计 1.5–3 h，可能要拆成两次 Kaggle 会话或减 epoch。未实测。

**报告的 COCO 增益** [已核实，论文表 2，5 折平均，AP50:95]
- YOLOv5l + Efficient Teacher：1% 23.76 / 2% 28.70 / 5% 34.11 / **10% 37.90（论文标注 +13.86）**，109.59G FLOPs。
- Dense Detector(ResNet50) 版：10% 34.09（+10.05），同架构监督 24.04；同架构重实现的 Unbiased Teacher 30.34（+6.30）。
- 同表其他方法 10%：Unbiased Teacher 31.50、Soft Teacher 34.04、LabelMatch 35.49、PseCo 36.06、Unbiased Teacher v2 32.61、DSL 36.22、Dense Teacher 37.13。
- **口径不一致要注意**：README 写 YOLOv5l 监督 10% = 28.45，SSOD = 37.9（即 +9.45）；论文的 +13.86 是相对 24.04（Dense Detector 监督）算的。引用时用 README 的 28.45 -> 37.9 更诚实。
- COCO-additional：49.0 -> 50.45（+1.45）。README：YOLOv5s 全量 37.2 -> 38.1（+0.9）。
- 没有 YOLOv5n/s 在 10% 标注下的官方数字 [未找到]。

**在新版 torch / Python 上的已知或可预期问题**
- 已核实的事实：`requirements.txt` 只有下界（torch>=1.9.0），无 Python 3.13 相关声明；代码搜索显示 11 个文件用 `torch.load(`，7 个文件 import `pkg_resources`，6 个文件用 `torch.cuda.amp`；搜索 `np.int` 命中 16 个文件（GitHub 代码搜索是分词匹配，可能包含 `np.int32` 这类合法用法，未逐行确认）。
- [判断] 预期要改的地方：(1) torch>=2.6 的 `torch.load` 默认 `weights_only=True`，加载整模型 pickle 的 checkpoint 会报错，需全部加 `weights_only=False`；(2) `pkg_resources` 在新 setuptools / Python 3.12+ 环境里可能不存在，需装 setuptools 旧版或改掉 `check_requirements`；(3) 若确有 `np.int`/`np.float` 别名，NumPy>=1.24 会报错；(4) `torch.cuda.amp.autocast/GradScaler` 目前只是弃用警告。这些都是 YOLOv5 时代代码的常见补丁，工作量估计 1–3 小时，但没有实测。
- issue 里反复出现的非安装类问题 [只核实标题]：SSOD 阶段 P/R/mAP 变 0（#64「用标准yolov5s转换为et模型后进行半监督训练，在训练过程中所有指标变为0」、#98、#155「训练lables/P/R都为0」、#161「关于半监督中指标不正确的情况」）、模型转换报错（#16、#23、#43、#105、#118）、#157「LabelMatch并没有调用，一直保持恒定阈值0.1 和 0.6」。**这是比安装更大的风险：小数据集上 SSOD 阶段崩溃、需要调阈值。** README 自己也说不建议直接开 SSOD，阈值与任务强相关。
- 许可：GPL-3.0（Ultralytics 是 AGPL-3.0）；只做实验对比没问题，公开合并代码时要注意。

---

## 3. 工业 / 表面 / PCB 缺陷上的 SSOD 与伪标签论文

| 论文 | 设置 | 对比的基线 | 数字 | 核实程度 |
|---|---|---|---|---|
| TaDP-Det, Sensors 2026, 26(4):1085 https://pmc.ncbi.nlm.nih.gov/articles/PMC12944570/ | Faster R-CNN R50，MMDetection 3.3.0；NEU-DET 1/5/10%，GC10-DET 5/10%，PCB-DEFECT 5/10% | Semi-DETR、Sparse Semi-DETR、STEP-DETR、DSL、Unbiased Teacher v2、ARSL、Consistent Teacher、Soft Teacher、PseCo、Mix Teacher；监督 DINO-DETR / FCOS / Faster R-CNN | PCB-DEFECT **10% mAP@0.5**：监督 Faster R-CNN 82.2 -> Soft Teacher 88.6、PseCo 90.8、Mix Teacher 90.1、UT v2 87.0（其监督 FCOS 73.0）、Consistent Teacher 80.2、Semi-DETR 91.1（监督 DINO 84.5）、TaDP-Det 92.7。5%：Faster R-CNN 66.8 -> Soft Teacher 83.1 -> TaDP-Det 89.8 | [已核实] 表 4（页面由小模型转述，个别数字建议引用前再对原表）；未见代码发布 |
| Ge et al., "Unsupervised selective labeling for semi-supervised industrial defect detection", J. King Saud Univ. CIS 36(8), 2024-10 https://www.sciencedirect.com/science/article/pii/S1319157824002684 | 以 **Efficient Teacher** 为基础，谱聚类选择要标注的样本；密封圈 / 金属表面 / PCB 三个数据集 | 摘要称在 10% 标注下把 Efficient Teacher 提升 5 / 6.6 / 7.8 mAP；搜索摘要称还比了 Faster R-CNN、YOLOv8、YOLOv11、Unbiased Teacher、Soft Teacher、Semi-DETR | 绝对值未取到（ScienceDirect 返回 403） | [摘要级]；基线清单 [未核实] |
| Wang et al., Sensors 2025, 25(23):7283 https://pmc.ncbi.nlm.nih.gov/articles/PMC12694290/ | 改进 YOLOv5；PKU-PCB，100 张有标 + 1000 张无标；**单轮离线伪标签自训练**（阈值 0.60，无 EMA 教师，无弱/强增强） | **没有任何 SSOD 基线的实验对比**，作者明确把与 teacher-student 方法的数值比较留作未来工作 | mAP@0.5：基线 0.870 -> +SSL 0.915；改进模型 0.910 -> +SSL 0.943 | [已核实] |
| Pham et al., "Defect Detection in PCBs Using Semi-Supervised Learning" (PCB_SS), Sensors 2023 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10051373/ | 半监督 PCB 检测 | 只知道"优于仅用有标数据的监督模型" | — | [摘要级] |
| Wan et al., DE-SSD, Sensors 2022 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9611715/ | 数据扩充策略 + 半监督 PCB | — | — | [摘要级] |
| DSYM, arXiv 2507.05588 | teacher-student + 条件扩散合成伪缺陷 + CLIP 过滤；NEU-DET | 摘要称优于已有半监督方法 | NEU-DET 78.4 mAP@0.5 | [摘要级] |
| Semi-SSDDet, Signal Image Video Process. 19(10):798, 2025 (Liu, Chu, Gong) | YOLO 系半监督，NEU-DET / GC10-DET，自适应伪标签阈值 | 未取到 | 未取到 | 只核实了 DBLP 条目 |
| He et al., ESANN 2020 https://www.esann.org/sites/default/files/proceedings/2020/ES2020-33.pdf | DeepPCB 上 EMA 教师 + 一致性损失 | — | — | [摘要级] |

观察：
1. 工业缺陷 SSOD 论文的基线惯例分两派：MMDetection 派（Soft Teacher / PseCo / Unbiased Teacher v2 / Consistent Teacher / Semi-DETR，全是 Faster R-CNN / FCOS / DETR 学生）和 YOLO 派（Efficient Teacher 几乎是唯一被引用的 YOLO-SSOD 基线）。
2. 发表在 Sensors 级别期刊上的 PCB 半监督论文（Wang 2025）**完全没有跑 SSOD 基线**也发表了；更高要求的审稿人会要 Efficient Teacher 或 mean-teacher 对比。
3. 10% 标注下的 SSOD 增益在这些数据集上是 mAP@0.5 +6 到 +9 个点（Faster R-CNN 82.2 -> 88.6–90.8）。我们的指标是 mAP50:95（0.352 -> 0.430，+0.078），不能直接横比，但量级相当。
4. 没有找到在 DsPCBSD+ 上做半监督或伪标签的论文 [搜索未命中，不等于不存在]。DsPCBSD+ 数据集论文：Scientific Data 2024, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11263390/（10,259 张图、20,276 个缺陷、9 类）。

---

## 4. 基础模型 / VFM 做伪标签器训练小检测器（新颖性判断）

| 工作 | 做法 | 数字 | 核实 |
|---|---|---|---|
| Lavoie et al., "Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection"（DINO Teacher），CVPR 2025，arXiv 2503.23220 | **冻结 DINOv2 骨干 + 只在源域有标数据上训练检测头，作为离线标注器**给目标域打伪标签；并把学生特征对齐到 DINO 编码器。明确声称标注器"比 Mean Teacher 生成更准确的标签" | 摘要只说多个 DAOD 数据集 SOTA | [摘要级]。**与我们最接近的先例**：区别在于它是域自适应、Faster R-CNN 学生，不是同域少标注、YOLO-nano 学生 |
| "Robust Object Detection with Pseudo Labels from VLMs using Per-Object Co-teaching"，arXiv 2511.09955 | VLM（搜索摘要说是 OWLv2，摘要页未写明）离线伪标签训练 YOLO，两个 YOLO 逐目标 co-teaching 滤噪 | KITTI mAP@0.5：YOLOv5m 基线 31.12 -> 46.61；**伪标签 + 10% 真值 = 57.97** | [摘要级] |
| Griffin et al., "Auto-Labeling Data for Object Detection"，arXiv 2506.02359 | 配置 VLM 基础模型生成伪标签，训练轻量检测器，完全不用真值 | 摘要无数字 | [摘要级]；PDF 过大未读 |
| VG-DETR，arXiv 2508.11167 | 半监督 DETR（无源域自适应，遥感）；VFM 用来筛选 / 找回伪标签并做特征对齐，在线教师仍是 EMA | 摘要无数字；搜索摘要称 VFM 是冻结的 DINOv2 ViT-L 且推理放在离线阶段 | [摘要级]，后半句 [未核实] |
| "Beyond Boundaries: Leveraging VFMs for Source-Free Object Detection"，AAAI，arXiv 2511.07301 | 把 Grounding DINO 的预测与 EMA 教师预测融合成伪标签；称只用其一都更差 | — | [摘要级] |
| "Training a Student Expert via Semi-Supervised Foundation Model Distillation"，arXiv 2604.03841 | 半监督蒸馏：先让 VFM 教师做域适应再冻结，蒸馏到约小 11 倍的学生；**实例分割**（Cityscapes、ADE20K） | 比零样本教师 +11.9 / +8.6 AP，比适应后教师 +3.4 / +1.5 AP | [摘要级]；搜索摘要称教师是 Grounding-DINO-L + SAM2-L，[未核实] |
| ICIP 2025 "Improving YOLOv8 for fast few-shot object detection by DINOv2 distillation" | 冻结 DINOv2 特征蒸馏到 YOLOv8，少样本检测 | — | 只见标题 / 搜索摘要 |
| Autodistill（工具，无论文） | 零样本基础模型标注 -> YOLOv8 | 无 | [已核实] |
| arXiv 2602.04699 航天器检测 | VLM 伪标签 + TTA / WBF -> 轻量学生 | — | [摘要级] |

新颖性判断（我的判断）：
- "基础模型当离线标注器训练小检测器"**本身不新**：DINO Teacher (CVPR'25) 已经做了"冻结 DINOv2 + 少量标注微调检测头 -> 离线伪标签 -> 胜过 mean teacher"，Autodistill 把流程产品化，2511.09955 做了 VLM -> YOLO。
- 我们可以主张的差异点：(a) 同域少标注（10%）而非域自适应或零样本；(b) 工业 PCB 缺陷这种文本提示型 VLM 基本无效的领域，用的是**在 10% 标注上微调过**的 DINOv2 检测器（RF-DETR），而不是零样本 OVD；(c) 学生是 nano 级实时模型；(d) 对照实验"同架构更大的 YOLOv8m 标注器 0.404 vs VFM 标注器 0.430"说明增益来自预训练表征而非单纯容量。没有找到用 RF-DETR 做伪标签器、或在 PCB / 表面缺陷上做 VFM 伪标签的已发表工作 [搜索未命中，不能断言不存在]。
- 应当引用并主动区分：DINO Teacher、2511.09955、Autodistill、Efficient Teacher；最好再做一次 Grounding DINO / OWLv2 零样本标注的对照（预期在 PCB 上很差），用来支撑 (b)。

---

## 5. 推荐（按优先级）

**第 1 位：在 Ultralytics 内自己实现 EMA mean-teacher（YOLOv8n 学生），必做。**
- 内容：有标数据 burn-in（可直接复用现有 0.352 的监督 checkpoint）-> 教师 = 学生 EMA（0.999 或按步数余弦）-> 教师看弱增强（仅翻转 / 缩放，无 mosaic）出伪标签，NMS + 置信度阈值（如 0.5–0.7，或 Efficient Teacher 式双阈值 0.1 / 0.6 把中间区间作不确定样本）-> 学生看强增强（mosaic + HSV + cutout / erasing）-> 无监督损失权重 1–4，有标 : 无标 批内 1:1。
- 工作量 [判断]：自定义 `DetectionTrainer` 子类 + 双 dataloader + 伪标签坐标随几何增强变换，1–2 天编码调试；每次训练约为监督训练的 2–2.5 倍时长，T4 上 60 epoch 应在 1–2 h 内（226px 小图有利）。风险：中。主要风险是伪标签阈值选择和 mosaic 下的框变换写错；Ultralytics 已自带 EMA，可借用。
- 优点：同一学生、同一代码库、同一 epoch / imgsz、同一评估——这是最干净的对照，也是唯一能回答"离线 VFM 标注器 vs 在线自教师"这个科学问题的实验。
- **审稿人是否接受它作为"Unbiased / Efficient-Teacher 式"基线**：我的判断是**可以接受，但措辞必须诚实**。应称为 "EMA mean-teacher baseline (weak-strong augmentation, confidence-thresholded online pseudo-labels), following the Unbiased Teacher / Efficient Teacher recipe, re-implemented in Ultralytics for YOLOv8n"，不能直接叫 "Unbiased Teacher" 或 "Efficient Teacher"——Unbiased Teacher 还包括 focal loss 处理类别不平衡、且不对框回归做无监督损失；Efficient Teacher 包括 Pseudo Label Assigner（双阈值软目标）和 Epoch Adaptor（域适应 + 分布自适应阈值）。先例：蓝莓基准 arXiv 2509.20580 就是在 Ultralytics YOLO 上用 "Unbiased Mean Teacher" 式实现并发表的（增益 −1.4 到 +2.9 mAP50，说明这个基线在 YOLO 上本来就不强，结果好坏都要如实报告）。要让它可信，需要：报告阈值敏感性（至少 3 个阈值）、多个种子、公开代码、说明与原方法的差异。只有这一个自实现基线时，挑剔的审稿人仍可能说"基线是自己实现的、可能没调好"，所以建议配第 2 位。

**第 2 位：Efficient Teacher 官方代码 + YOLOv5n（或 v5s）学生，强烈建议做。**
- 理由：唯一有论文、有官方代码、被缺陷检测文献当基线引用（Ge et al. 2024）的 YOLO 系 SSOD；审稿人认识它。
- 代价 [判断]：环境补丁 1–3 h（`torch.load` 的 weights_only、`pkg_resources`、可能的 numpy 别名）；自己写 yolov5n 的 SSOD yaml；数据列表转换很简单。每次 60 epoch 估计 1.5–3 h，可能超出单次预算，需减 epoch 或分两段。风险：**中到高**——issue 里大量"SSOD 阶段指标变 0"的报告，小数据集上阈值要调；作者不再维护。预留 2–3 天，并准备好"跑不通就如实写明尝试过"的退路。更稳的做法是另建 Python 3.10 + torch 1.13 / 2.0 的环境而不是硬上 3.13 / 2.11。
- 注意混杂因素：学生是 YOLOv5n（anchor-based，1.87M 参数）而不是 YOLOv8n，所以必须在 Efficient Teacher 代码库里同时跑**监督 YOLOv5n**，报告"相对各自监督基线的增益"，而不是绝对 mAP 横比。更完整的做法是把我们的 RF-DETR 离线伪标签也喂给 Efficient Teacher 代码库里的 YOLOv5n 监督训练，得到同学生下的对比。

**第 3 位（低成本补充，建议做）：经典离线自训练 / STAC 式基线 + 零样本 VFM 对照。**
- 自训练：YOLOv8n 自己给无标数据打标再重训（学生 = 标注器）。配合已有的 YOLOv8m 标注器（0.404），形成"自标注 -> 更大同构标注器 -> VFM 标注器"的阶梯。成本：1 次推理 + 1 次训练，几乎零新代码。
- Grounding DINO / OWLv2 零样本（Autodistill 式）标注 -> YOLOv8n：用来说明为什么需要微调过的 VFM 检测器。成本：半天；T4 上推理 7.4k 张小图可行。

**不建议做**：Soft Teacher / PseCo / Unbiased Teacher v2 / Consistent-Teacher / Semi-DETR / MixPL。原因：(1) 仓库全部停更或归档，依赖 MMDet 2.x / mmcv 1.x / Detectron2 / cvpods，在当前环境下安装成本高且可能失败；(2) 学生是 Faster R-CNN R50 / FCOS / DINO-DETR，比 YOLOv8n 大一个数量级，与"小检测器"问题不可比；(3) 原始训练计划是 8 卡 180k 迭代级别。处理方式：在相关工作里引用，并在正文说明因学生架构 / 计算量不可比而未纳入；如审稿人坚持，最可行的是 MMDet 3.x 内置的 SoftTeacher 配置 + 独立 py3.10 环境，预计 2–4 天，风险高。

**建议的最终对比表（均为 10% 标注，≥3 个种子）**：监督 YOLOv8n(0.352) / 自训练 YOLOv8n / YOLOv8m 标注器(0.404) / Ultralytics 内 mean-teacher YOLOv8n / [Efficient Teacher YOLOv5n 及其监督基线，单列] / 零样本 OVD 标注器 / RF-DETR 标注器(0.430) / 全监督 100% 上界。另一个值得做的组合：mean-teacher 从 RF-DETR 伪标签训练出的学生起步（两者是否互补），这往往是审稿人的下一个问题。

---

## 没查到 / 没做到的事（如实列出）
- 没有在 Python 3.13 / torch 2.11 上实际安装或运行任何仓库。
- Efficient Teacher 没有 YOLOv5n/s 的 10% 官方数字；issue 只看了标题没读回复。
- Ge et al. 2024 的绝对数字和完整基线清单未取到（403）；Semi-SSDDet、SUD-YOLO、DSYM、PCB_SS、DE-SSD 只到摘要级。
- arXiv 2506.02359 和 2511.09955 用的具体 VLM 与学生模型未从正文确认。
- GitHub 仓库 API 在本会话被限制，仓库信息来自搜索 API（pushed_at、许可、star、归档状态）和 raw 文件；"最后 commit 日期"未逐条核对。
- 未检索中文数据库（知网等）里的 PCB 半监督工作。
