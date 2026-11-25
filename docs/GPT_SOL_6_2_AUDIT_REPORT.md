# sqlite-vec-hybrid 深度代码审计与复验

审计日期 **2026-11-25**（台北时间）。三层基准裁决为 **PASS WITH WARNINGS**。两套 Windows x64 Release 全量重编均为 **0 警告、0 错误**，CTest 各 **38/38**。FP32 HNSW 在 20K 合成向量上 Recall@10 为 **0.9935**，5% 预过滤为 **1.0000**。可选 SQ8 的 Recall@10 为 **0.9625**，低于 0.99。

审计覆盖公开 C 头文件、数值与图索引、SQLite 静态和动态入口、虚表写入与事务生命周期、查询规划和 FTS5、隐私扫描、CLI、基准脚本、构建与测试。下文行号指向修复后的文件。风险分级依据可到达的内存安全、持久化一致性和数据完整性影响。

## 1. 分级发现与修复

### Critical · 0 项

本次审计确认的最高风险等级为 High。

### High · 8 项

| 编号 | 文件与行号 | 根因及影响 | 修复与回归 |
| --- | --- | --- | --- |
| H1 | `src/sve_hnsw.c:1259,1355,1387`；`src/sve_vtab_guard.c:1054` | CRC 后缺少完整图结构及虚表配置校验。节点数和容量计算的边界不足，跨层边和不匹配的配置可能引发越界访问。 | 校验固定记录下界和乘法、两阶段验证前向边、层级、度上限、自环、重复边/rowid、存活计数、入口和全局最高层。检查向量编码、CRC、魔数和版本。v2 只写存活邻接，合法 v1 的历史墓碑边在加载时清理。虚表在发布前核对快照配置。回归 `audit_vtab_snapshot_connect`、`audit_hnsw_legacy_snapshot`、`hnsw_snapshot_crc_corrupt`。 |
| H2 | `src/sve_hnsw.c:695,804,835` | 原插入提前增加节点数并发布映射，后续搜索堆、候选及反向边临时分配失败会留下部分图状态。 | 先在未发布节点上准备邻接和容量，最后更新映射并无分配地发布反向边。失败释放暂存对象并恢复 RNG。分配器逐点注入 18 个失败，失败前后快照字节一致，重试与首次成功一致，索引分配计数归零。回归 `audit_hnsw_insert_oom`。 |
| H3 | `src/sve_vtab_guard.c:1022,1349,2267,2315,2343,2430` | 旧行先删、独立影子写入、失败补偿与 SQLite 语句日志时序不一致。单条 VALUES 写入在显式事务中可能没有外层语句日志，ROLLBACK TO 原本清除快照待写状态。 | 约束前置，内层事务保护语句统一覆盖 `_rows`、`_fts` 与图修改。失败由 SQLite 回滚。按保存点状态延迟重建，提交时写快照并传递绑定失败。回归 `audit_vtab_shadow_write_failure`、`audit_vtab_hnsw_insert_failure`、`audit_vtab_savepoint_conflicts`。 |
| H4 | `src/sve_vtab_guard.c:966,1095,1353` | 多连接各自缓存图，另一连接提交后可能漏读，后续写入还可能把旧图写回。 | 在同一 pager 读视图中使用 schema `data_version` 刷新完整快照。回滚将缓存标记为待恢复，重建成功后替换旧图。WAL 双连接回归 `audit_vtab_connection_visibility`。 |
| H5 | `src/sve_vtab_guard.c:675,711,1131,1156,1304` | 错误路径使用已释放的 DDL 参数。影子 DDL 静默接管同名对象，销毁忽略 SQLite 错误。 | 生成不引用已释放参数的诊断，精确解析选项并安全引用标识符。创建拒绝冲突对象，删除传播错误，明确拒绝未实现的重命名。回归 `audit_vtab_shadow_ownership_identifiers`。 |
| H6 | `src/sve_numeric.c:431,497` | 原解析器将有长度的输入切片直接传给 C 字符串转换函数，转换可越过切片末尾，逗号小数 Locale 也会干扰分隔符。 | 有界 ASCII JSON 语法检查，浮点转换前把小数点移入指数，int8 逐位限幅。拒绝尾随非分隔符、孤立符号及其他非法数字。两套 CRT 均用逗号小数 Locale 验证。回归 `audit_json_locale_and_numeric_extremes`。 |
| H7 | `cli/sve_hybrid_cli.c:369,683,747` | ANN 分配尺寸可回绕，TSV 相关 ID 的分配与读取使用不同分隔规则，且部分扩容和 SQL 错误未检查。 | 严格解析参数、记录和分隔符，先检查乘法和容量，再分配。验证向量、相关 ID、入库与查询返回，失败统一清理。回归 `audit_cli_input_validation`、`cli_audit_roundtrip`。 |
| H8 | `src/sve_sqlite_ext.c:632,719,729` | DLL 注册在半途失败后，已安装回调可能继续指向由 SQLite 关闭的模块。文件名默认加载入口也与导出名不一致。 | 注册前拒绝活动 VM，在连接互斥保护下撤销失败前的注册。保留显式入口并增加文件名推导入口，动态加载要求 SQLite 3.45.0+。回归 `audit_abi_static_subtypes`、`audit_abi_loadable_subtypes`、`ext_loadable_dll`。 |

有关 SQLite 虚表约束政策和扩展 subtype，核对了捆绑的 SQLite 3.45.3 实现及官方接口契约。[虚表约束政策](https://www.sqlite.org/c3ref/c_vtab_constraint_support.html) · [结果 subtype](https://www.sqlite.org/c3ref/result_subtype.html) · [参数 subtype](https://www.sqlite.org/c3ref/value_subtype.html)

### Medium · 6 项

| 编号 | 文件与行号 | 根因及影响 | 修复与回归 |
| --- | --- | --- | --- |
| M1 | `src/sve_hnsw.c:952,1103` | 删除后保留向量和链接缓冲。ef 乘法提升可回绕，断图和大 k 遇到候选不足时结果可能短缺。 | 及时释放已删节点的向量和链接，重选最高存活入口，有界提升 ef，候选不足或超过 ef 时走对应精确搜索。回归 `audit_hnsw_entry_allowlist`。 |
| M2 | `src/sve_vtab_guard.c:1378,1580,2044` | 旧计划的 16 条元数据上限提前发生在 128 字节布局之前，剩余过滤被 SQLite 放到 top-k 之后。排序规则也可能丢失。SQLite 3.45 可在长计划中保留 MATCH 残余判定。 | 动态生成受控谓词、完整绑定，保留 SQLite collation。残余 MATCH 核对当前行和已执行的原始 MATCH 参数。回归 `audit_vtab_metadata_plan`，含超过 45 条 AND 谓词和 NOCASE。 |
| M3 | `src/sve_vtab_guard.c:1493,1632,1698` | 重复 MATCH 静默覆盖，数值控制被隐式转换，NaN/NULL 可误选融合分支。词法候选在元数据过滤前限量会漏掉有效候选。 | 校验唯一控制项、k 的整数范围、alpha 有限性，NULL 文本条件产生空结果。相同元数据条件在 BM25 LIMIT 前应用，游标重入释放所有结果及绑定。回归 `audit_vtab_query_controls`、`audit_vtab_metadata_plan`。 |
| M4 | `src/sve_vtab_guard.c:44,72,169,303` | PEM 反复扫描长行并误接受其他尾标。相邻凭据、带符号的引用值、长 DSN 凭据可能脱敏不全。 | 识别有界 PEM 头和匹配尾标，未闭合块脱敏至末尾。完整处理引用及逃逸、相邻前缀和 DSN 边界，并在分配前限制文本长度。回归 `audit_privacy_scan_boundaries`、`privacy_ood_15_15`。 |
| M5 | `src/sve_hybrid_rank.c:89,219,294,329` | 清洗器先执行无界长度扫描，接受截断的 UTF-8。融合输入长度和浮点极差可能回绕或溢出。 | 有界 UTF-8 校验和整词输出，保留字作为字面词。检查融合尺寸，在必要时缩放极值跨度。回归 `audit_rank_and_fts_utf8`、`hybrid_fts5_sanitizer`。 |
| M6 | `cli/sve_hybrid_cli.c:152,478`；`benchmarks/run_benchmarks.py:118,189,278` | 执行错误可被统计为成功或无误报。旧 runner 无条件输出 verified，并在数据目录写派生 TSV。 | 核对原始比较结果、脱敏残留和实测计数，独立记录门槛与裁决。派生 TSV 写工作区的临时目录，原始数据文件包含 SHA-256 以便复查。回归 `audit_cli_input_validation`、`cli_audit_roundtrip` 和三层基准。 |

### Low · 2 项

| 编号 | 文件与行号 | 根因及影响 | 修复与回归 |
| --- | --- | --- | --- |
| L1 | `src/sve_numeric.c:40,246,292` | SQ8 的非零极小尺度可能舍入为零，解量化可能成功返回非有限值。通用字节换算存在回绕边界。 | 保留可表示的正尺度，在 double 范围检查重建值，再转为 float。保护字节换算。回归 `audit_json_locale_and_numeric_extremes`、`numeric_crc32c_and_sq8`。 |
| L2 | `src/sve_sqlite_ext.c:146,288` | 维度错误在释放后读取了已清零计数，内存失败丢失状态，JSON 结果构造绕过连接长度上限。 | 释放前保存维度诊断，保留 NOMEM 状态，构造器使用当前连接的资源上限。bit BLOB 先比较字节再乘 8。回归 `audit_abi_static_subtypes`、`audit_abi_loadable_subtypes` 及原有标量测试。 |

float32 的 L2 和 Cosine 保持 double 累加。新增回归覆盖 `3e38`、`1e-44`、反向极值与混合量级。动态和静态入口都测了 subtype 的产生、嵌套消费、虚表列输出和表达式索引。

## 2. 上游及公共依赖零改动

对审计前后的每个文件读取原始字节。按相对路径排序后，把相对路径、字节数与文件 SHA-256 汇总为目录 SHA-256。

| 目录 | 文件数 | 总字节数 | 前后相同的 SHA-256 | 改动字节 |
| --- | ---: | ---: | --- | ---: |
| `upstream/sqlite-vec` | 325 | 2,850,566 | `c30e1435549ca5206a495f10fda6051d1bad40f9f642004474b346dd3bcb3bac` | 0 |
| `third_party/sqlite3` | 4 | 10,619,592 | `701b5a196c3500d07d04182e06aefeb8f6901a1df5a9ca0f41d0f4883bf7fae6` | 0 |

`tests/test_upstream_repro.c` 的审计前后 SHA-256 均为 `f04aadde23c7e771f4780bedc6b8870c0ccd119f96a84ffb6f4a1684b7c63011`。双编译器原有 21 项均通过，包括 `#333`、`#361`、`#334`、`#362/#363`。基线源码标注为 `asg017/sqlite-vec@04d28bd21773981e2d266bbf6aa4efbd011eb4f6`，SQLite 为 3.45.3。

## 3. 双编译器全量复验

| 工具链 | 自有目标编译选项 | 警告 / 错误 | CTest |
| --- | --- | ---: | ---: |
| MSVC 19.50.35725.0 x64 Release | `/std:c11 /W4 /WX` | 0 / 0 | 38/38 |
| MinGW-W64 GCC 16.2.0 x64 Release | `-std=c11 -Wall -Wextra -Wpedantic -Werror` | 0 / 0 | 38/38 |

各 38 项由原有 21 项及新增 17 项组成。项目目标以警告为错误。两处供应商源码沿用 CMake 原有的 `/W0` 或 `-w`。两套测试分别覆盖了逗号小数点 Locale 与 18 个插入分配故障点。本地回归日志输出于 `build/audit-msvc-build.log`、`build/audit-msvc-ctest.log`、`build-gcc/audit-gcc-build.log`、`build-gcc/audit-gcc-ctest.log`。

## 4. 三层基准实测

运行记录生成于 `2026-11-25T06:27:00.665225+00:00`（UTC）。计时为本机单次 CPU 运行，质量指标采用固定数据和参数。机器可读结果见 [benchmarks_report.json](benchmarks_report.json)。

### Tier 1 · BEIR 公开测试切分

仅使用语料拟合 128 维 TF-IDF/SVD，二值相关性按 test qrels 正分判定。两组查询与语料计数均经 CLI 实际处理交叉核对。

| 数据集 | 模式 | nDCG@10 | MRR@10 | Recall@10 | P50 (ms) | P99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciFact（5,183 / 300） | vec0 FLAT | 0.3815 | 0.3372 | 0.5336 | 1.7484 | 3.4474 |
| SciFact | sve0 FP32 HNSW | 0.3801 | 0.3364 | 0.5302 | 0.4933 | 0.9806 |
| SciFact | FTS5 BM25 | 0.6683 | 0.6348 | 0.7956 | 10.0900 | 39.1692 |
| SciFact | 混合 RRF（融合常数 60） | 0.5401 | 0.5008 | 0.6938 | 10.1354 | 42.9128 |
| SciFact | 混合 Convex（α=0.25） | 0.6714 | 0.6380 | 0.8024 | 10.2339 | 36.8949 |
| NFCorpus（3,633 / 323） | vec0 FLAT | 0.2429 | 0.3859 | 0.1126 | 1.3688 | 3.2338 |
| NFCorpus | sve0 FP32 HNSW | 0.2420 | 0.3828 | 0.1124 | 0.4875 | 1.1415 |
| NFCorpus | FTS5 BM25 | 0.3065 | 0.5124 | 0.1487 | 0.9037 | 10.0984 |
| NFCorpus | 混合 RRF（融合常数 60） | 0.2843 | 0.4377 | 0.1521 | 1.4408 | 11.3301 |
| NFCorpus | 混合 Convex（α=0.25） | 0.3240 | 0.5315 | 0.1658 | 1.4582 | 11.9077 |

### Tier 2 · 20,000 向量 ANN 与上游对比

固定种子 42，128 维、200 个查询，k=10。5% 预过滤命中 1,000 行，召回基准为精确搜索。

| 模式 | 构建 (ms) | P50 (ms) | P99 (ms) | P50 × | Recall@10 | 过滤 P99 (ms) | 过滤 Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vec0 FLAT | 116.29 | 3.1976 | 4.5165 | 1.00 | FLAT | 4.1088 | — |
| FP32 HNSW | 28770.15 | 0.4935 | 1.0906 | 6.48 | 0.9935 | 2.2687 | 1.0000 |
| SQ8 HNSW | 10645.04 | 0.3902 | 0.7724 | 8.19 | 0.9625 | — | — |

构建复用 visited 和候选堆，反向边选择用有界栈缓冲，余弦范数只在向量载入或查询开始时计算。在相同 ANN 参数下，审计前 MSVC FP32 HNSW 测得 **61,045.29 ms**，本轮全量复跑为 **28,770.15 ms**，观测降幅 **52.9%**。二者 Recall@10 均为 **0.9935**。构建计时遵循 CLI 从批量写入至提交的口径，含快照写入。单次计时会受系统负载影响。

上游对比矩阵实测 `#333` 上游 1 条、混合 0 条，`#361` 上游拒绝、混合接受，`#334` 上游出现 Inf/0、混合返回 `1e20` 和 `1e-25`，`#363` 上游回绕、混合拒绝。`#362` 由原始 CTest 独立覆盖。

### Tier 3 · 固定本地隐私样本

敏感样本命中 **15/15**，良性误报 **0/15**。补充 CTest 覆盖相邻凭据、未闭合 PEM、256 KiB 长文本及 IP、版本号、时间戳和 Trace ID。该结果对应固定样本。

### 最终裁决 · PASS WITH WARNINGS

FP32 的无过滤与过滤 Recall@10、上游矩阵和隐私样本门槛全部通过。SQ8 的 Recall@10 为 0.9625，这是本轮告警。新写快照为 SVEH v2，打开它的客户端需支持 v2。现有合法 v1 快照可读。组合范围谓词和标量等值谓词支持 top-k 前过滤，SQLite 对未下推的 IN/OR 表达式继续处理残余条件。图断开且返回候选不足时回退精确检索，候选足量时保留近似搜索。

## 5. 复验命令

在工作区使用 MSVC x64 开发者终端和 MinGW-W64 环境分别构建。在环境变量 `SVE_BENCHMARK_DIR` 中指定本次提供的外部数据目录。

```powershell
cmake --build build --config Release --clean-first --verbose
ctest --test-dir build --output-on-failure -V
cmake --build build-gcc --config Release --clean-first --verbose --parallel 4
ctest --test-dir build-gcc --output-on-failure -V
python benchmarks/run_benchmarks.py --cli build/sve_hybrid_cli.exe --dataset-dir $env:SVE_BENCHMARK_DIR --output docs/benchmarks_report.json --dims 128
```
