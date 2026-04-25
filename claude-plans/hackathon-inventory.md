# vLLM Hackathon — Track 6: Performance Tuning & Evaluation
## Repo Inventory & Reusable Components

---

## Repos Explored

| # | Repo | Path | Type |
|---|------|------|------|
| 1 | AI-Analysis-Agent | `research repo/AI-Analysis-Agent-master/` | LangGraph agent + 28 MCP tools |
| 2 | ai-perf-hackathon | `research repo/ai-perf-hackathon-main/` | Autonomous Claude agent for RHEL/Nginx tuning |
| 3 | LLM-inference-benchmark-guide | `research repo/LLM-inference-benchmark-guide-main/` | Documentation only — benchmarking methodology |
| 4 | performance-dashboard | `research repo/performance-dashboard-main/` | Streamlit dashboard (RHAIIS / LLM-D / MLPerf) |
| 5 | vllm-profiler | `research repo/vllm-profiler-main/` | K8s webhook for automatic vLLM PyTorch profiling |

---

## Reusable Components by Category

### 1. Agentic Loop / Orchestration

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| LangGraph ReAct agent with tool calling, SSE streaming, thread persistence | AI-Analysis-Agent | `psap-agent/psap_agent/src/core/agent.py`, `manager.py` | Uses Google Gemini. `AsyncPostgresSaver` for checkpointing. Langfuse callbacks for tracing. |
| Autonomous tool-use loop with Claude API | ai-perf-hackathon | `agent/agentic.py` | Claude tool_use format. Tools: `run_command`, `read_file`, `write_file`, `run_benchmark`, `done`. Iteration-limited. |
| Tool definition framework (JSON schema for Claude) | ai-perf-hackathon | `agent/tools.py` | `ToolResult` dataclass. Command history tracking. SSH-based execution. |
| System prompt with structured triage workflow | AI-Analysis-Agent | `psap-agent/psap_agent/src/core/prompt.py` | Comprehensive tool usage instructions for performance analysis. |
| System prompt with tuning knowledge base | ai-perf-hackathon | `agent/agentic.py` (lines 25-181) | Nginx/RHEL tuning priorities, benchmark rules, NIC discovery workflow. |

### 2. MCP Tools (28 tools)

All from **AI-Analysis-Agent** — `psap-mcp-server/psap_mcp_server/src/tools/`

| Category | Tools | File |
|----------|-------|------|
| Benchmark query & comparison | `query_performance_metrics`, `compare_configurations`, `compare_versions_comprehensive` | `query_performance_tool.py`, `compare_performance_tool.py` |
| Discovery | `get_dataset_metadata`, `discover_configurations` | `dataset_metadata_tool.py`, `discover_configurations_tool.py` |
| Cost & regression | `calculate_cost_efficiency`, `analyze_regression` | `cost_efficiency_tool.py`, `regression_analysis_tool.py` |
| Grafana GPU metrics | `query_grafana_metrics`, `compare_grafana_metrics`, `generate_grafana_url` | `grafana_metrics_tool.py`, `compare_grafana_metrics_tool.py`, `generate_grafana_url_tool.py` |
| Dashboard URL | `generate_dashboard_url` | `generate_dashboard_url_tool.py` |
| PyTorch profiler | `analyze_pytorch_profile`, `compare_pytorch_profiles`, `analyze_performance_insights`, `list_available_profiles`, `check_profile_status` | `pytorch_profile_tool.py` |
| Kernel-to-code mapping | `map_kernel_to_vllm_code`, `get_kernel_categories`, `correlate_kernel_with_changes` | `kernel_code_mapper_tool.py` |
| vLLM source & diffs | `fetch_vllm_source`, `get_vllm_code_diff` | `kernel_code_mapper_tool.py` |
| vLLM releases | `get_vllm_release_notes`, `compare_vllm_versions`, `get_version_mappings`, `get_vllm_pull_request` | `vllm_release_notes_tool.py` |
| vLLM logs | `fetch_vllm_logs`, `compare_vllm_logs` | `vllm_log_tool.py` |
| Triage guide | `get_vllm_performance_triage_guide` | `vllm_performance_triage_tool.py` |

### 3. Metrics Collection

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| Benchmark data loader (S3 + local CSV fallback, TTL cache) | AI-Analysis-Agent | `psap-mcp-server/.../tools/performance_data_loader.py` | Pandas-based filtering. Loads RHAIIS/MLPerf/LLM-D CSVs. 5-min cache. |
| S3 utilities | AI-Analysis-Agent | `psap-mcp-server/.../tools/s3_utils.py` | boto3 client wrapper. |
| System metrics collection (CPU, memory, sysctl, NIC) | ai-perf-hackathon | `agent/collector.py` | SSH-based. `SystemMetrics` and `BenchmarkResult` dataclasses. |
| Grafana DCGM/vLLM metric queries | AI-Analysis-Agent | `psap-mcp-server/.../tools/grafana_metrics_tool.py` | GPU utilization, memory, power, temperature, clock speeds. Auto-discovers cluster from UUID. |
| Dashboard data loader with S3 + local fallback | performance-dashboard | `dashboard.py` — `load_data()`, `read_csv_from_s3()` | Streamlit `@st.cache_data` with 5-min TTL. |

### 4. Benchmarking & Workload Profiles

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| 4 ISL/OSL workload profiles (Balanced, Decode-Heavy, Prefill-Heavy, Long-Context) | LLM-inference-benchmark-guide | `README.md` | Standardized ISL/OSL pairs: 1000/1000, 512/2048, 2048/128, 8000/1000. |
| 7 concurrency levels | LLM-inference-benchmark-guide | `README.md` | `1, 50, 100, 200, 300, 500, 650` |
| GuideLLM command templates per profile | LLM-inference-benchmark-guide | `README.md` | OpenAI HTTP backend, concurrent rate type, 600s max duration. |
| vLLM runtime parameters (GPU, TPU, Spyre) | LLM-inference-benchmark-guide | `README.md` | `max-model-len`, `gpu-memory-utilization`, `tensor-parallel-size`, etc. |
| Model inventory (dense + MoE, BF16 + FP8) | LLM-inference-benchmark-guide | `README.md` | Llama 70B/405B, DeepSeek-R1, Qwen3-235B, GPT-OSS-120B, etc. |
| Benchmark execution + JSON result parsing | ai-perf-hackathon | `agent/collector.py` — `run_all_benchmarks()`, `get_latest_results()` | 5 workloads (homepage, small, medium, large, mixed). |
| Hardware platform specs (H200, MI300X, TPU v6e, Spyre) | LLM-inference-benchmark-guide | `README.md` | Memory, bandwidth, interconnect details. |

### 5. PyTorch Profiling

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| K8s mutating webhook for zero-code profiler injection | vllm-profiler | `webhook.py` | Flask-based. Multi-label OR selectors. Pod annotation-to-env conversion. JSONPatch mutation. |
| Python import hook (sitecustomize.py) wrapping `Worker.execute_model` | vllm-profiler | `sitecustomize.py` | `sys.meta_path` finder. Wraps with `torch.profiler`. Multi-range support. Chrome trace export. |
| Profiler config (YAML + env vars + annotations) | vllm-profiler | `profiler_config.yaml`, `CONFIGURATION_EXAMPLES.md` | Priority: defaults < YAML < env vars < pod annotations. |
| Chrome trace analysis (kernel stats, pipeline breakdown) | AI-Analysis-Agent | `psap-mcp-server/.../tools/pytorch_profile_tool.py` | Parses trace JSON from S3. Functional grouping (MoE, attention, quantization, etc.). Cross-version comparison. |
| Kernel-to-source mapping (120+ patterns) | AI-Analysis-Agent | `psap-mcp-server/.../tools/kernel_code_mapper_tool.py` | Maps kernel names to vLLM Python/CUDA source paths. GitHub API for live code + diffs. |
| Deployment automation (deploy, teardown, cert gen, validation) | vllm-profiler | `deploy.sh`, `teardown.sh`, `gen-certs.sh`, `validate_webhook.sh` | 8-step deployment. TLS cert generation. Comprehensive validation checks. |

### 6. Dashboards & Visualization

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| RHAIIS dashboard (throughput, latency, cost, regression, Pareto) | performance-dashboard | `dashboard.py` | Streamlit + Plotly. Geometric mean aggregation. Version comparison. |
| LLM-D dashboard (disaggregated prefill/decode) | performance-dashboard | `llmd_dashboard.py` | Prefill/decode pod counts. Efficiency ratio per TP unit. Profile assignment. |
| MLPerf dashboard (v5.0/v5.1 submissions) | performance-dashboard | `mlperf_datacenter.py` | Multi-row CSV header parsing. UTF-8/UTF-16 support. |
| IntelliConfig wizard (guided vLLM config generation) | performance-dashboard | `intelliconfig.py` | Model/accelerator/workload selection. Generates bare-metal CLI + OpenShift YAML. |
| CSS theming (light/dark, Red Hat branded) | performance-dashboard | `dashboard_styles.py` | 2,027 lines. KPI cards, responsive grids, color palettes. |
| Streamlit chat UI for agent | AI-Analysis-Agent | `psap-agent/examples/streamlit_app.py` | Multi-turn conversation. Feedback buttons. SSE streaming. |
| Grafana dashboard link generation | AI-Analysis-Agent | `psap-mcp-server/.../tools/generate_grafana_url_tool.py` | Pre-applied filters for DCGM/vLLM metrics. |

### 7. Cost & Regression Analysis

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| Cost-per-million-tokens with SLO constraints | AI-Analysis-Agent | `psap-mcp-server/.../tools/cost_efficiency_tool.py` | ITL P95 <= 65ms, TTFT P95 <= 3400ms. H200/MI300X/TPU pricing. |
| Version regression detection | AI-Analysis-Agent | `psap-mcp-server/.../tools/regression_analysis_tool.py` | Automated version-to-version comparison. |
| Performance comparison (geometric mean, peak, percentiles) | performance-dashboard | `dashboard.py` — `compare_two_datasets()` | Neutral threshold +/-2%. Concurrency intersection logic. |
| Token usage tracking + cost reporting | ai-perf-hackathon | `agent/llm.py` | Per-model pricing. Markdown table output. Multi-model support (Sonnet/Opus/Haiku). |

### 8. Observability & Tracing

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| Langfuse v3 integration (traces, sessions, cost) | AI-Analysis-Agent | `psap-agent/psap_agent/src/core/manager.py` | CallbackHandler on every LangGraph invocation. ClickHouse + Redis + MinIO stack. |
| Structured logging (structlog) | AI-Analysis-Agent | `psap-mcp-server/psap_mcp_server/utils/pylogger.py` | JSON output. Context tagging (run_id, thread_id, session_id). |

### 9. Deployment & Infrastructure

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| OpenShift manifests (9 pods: PG, Langfuse stack, MCP, Agent, Streamlit) | AI-Analysis-Agent | `openshift-manifests/01-06*.yaml` | Secrets, PVCs, deployments, services, routes. |
| Local dev Podman stack | AI-Analysis-Agent | `test-local-containers.sh` | Full stack in Podman containers. `rebuild`/`restart` modes. |
| Dockerfile (OpenShift-compliant) | performance-dashboard | `Dockerfile.openshift` | Python 3.12-slim. Non-root user (1001). Health check. |
| OpenShift deploy manifests for dashboard | performance-dashboard | `deploy/openshift-deployment.yaml`, `openshift-service.yaml`, `openshift-route.yaml` | Ready-to-apply. |
| K8s webhook deployment automation | vllm-profiler | `deploy.sh`, `manifests.yaml`, `kustomization.yaml` | Namespace, ServiceAccount, MutatingWebhookConfiguration. |
| CI/CD pipeline (GitHub Actions) | performance-dashboard | `.github/workflows/ci.yml` | 4 parallel jobs: lint, type check, test+coverage, docs check. |

### 10. Data Import & Processing

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| GuideLLM JSON to CSV converter | performance-dashboard | `manual_runs/scripts/import_manual_runs_json_v2.py` | guidellm 0.5.x format. 44-column output. TP and DP support. |
| MLPerf CSV parser (multi-row headers) | performance-dashboard | `mlperf_datacenter.py` — `load_mlperf_data()` | Handles UTF-8/UTF-16. Flattens complex headers. |
| Dataset summary generation | performance-dashboard | `datasets/generate_summaries.py`, `mlperf-data/original/generate_dataset_summaries.py` | Token length distributions for synthetic datasets. |
| S3 profiler trace upload | AI-Analysis-Agent | `scripts/upload-profiles-to-s3.sh` | Expected layout: `s3://<bucket>/profiles/rhaiis/<accelerator>/<model>/<version>/trace_rank<N>_*.json` |
| Version mappings (RHAIIS -> vLLM) | AI-Analysis-Agent | `psap-mcp-server/.../tools/version_mappings.json` | RHAIIS-3.1 through RHAIIS-3.3. |
| Consolidated benchmark CSV | AI-Analysis-Agent, performance-dashboard | `consolidated_dashboard.csv` | Local fallback when S3 unavailable. |

### 11. Remediation & Safe Tuning

| What | Source | File(s) | Notes |
|------|--------|---------|-------|
| Apply recommendations with backup + rollback | ai-perf-hackathon | `agent/remediator.py` | `RemediationAction` with rollback commands. nginx -t validation. Service reload. Dry-run mode. |
| LLM-based root cause analysis | ai-perf-hackathon | `agent/analyzer.py` | `TuningRecommendation`, `Bottleneck`, `AnalysisResult` dataclasses. JSON-structured LLM output. |
| Performance reporting (markdown + JSON) | ai-perf-hackathon | `agent/reporter.py` | Before/after comparison. Improvement % calculation. Token usage table. |
| SSH automation for remote systems | ai-perf-hackathon | `agent/ssh_client.py` | `SSHResult` dataclass. File I/O via heredoc. BatchMode. |
| Tuning knowledge base | ai-perf-hackathon | `config/tuning_rules.yaml` | Nginx, kernel, disk, network categories with check commands and optimal values. |

---

## Things We Still Need Clarity On

### Architecture & Scope

- **Which LLM powers the agentic loop?** AI-Analysis-Agent uses Google Gemini; ai-perf-hackathon uses Claude. Which do we target for the hackathon submission — or do we swap in a vLLM-served model?
- **What is the "agentic loop" doing exactly?** Is it: (a) automatically sweeping vLLM configs and benchmarking, (b) analyzing existing benchmark data and recommending changes, or (c) both?
- **Is vLLM the system under test, or also the LLM serving the agent?** The track says "vLLM end-to-end" — does this mean the agent should also run on vLLM?

### Benchmarking

- **Which benchmark client?** LLM-inference-benchmark-guide references GuideLLM. AI-Analysis-Agent references data already in S3/CSV. Do we run benchmarks ourselves or consume existing data?
- **Concurrency levels and profiles** — do we use the 4 profiles and 7 concurrency levels from LLM-inference-benchmark-guide, or define custom ones?
- **Target models and hardware** — which models and accelerators are available for the hackathon? The repos reference H200, MI300X, TPU, Spyre.

### Infrastructure

- **Prometheus/Grafana availability** — AI-Analysis-Agent's Grafana tools assume a running Grafana instance with DCGM exporters. Is this available in the hackathon environment?
- **S3 bucket access** — multiple repos assume AWS S3 for benchmark data and profiler traces. Do we have access, or do we work with local CSVs only?
- **OpenShift vs local** — are we deploying on OpenShift/K8s, or running everything locally?
- **PostgreSQL for state** — AI-Analysis-Agent uses PG for conversation persistence. Is this required, or can we use `InMemorySaver`?

### Profiling

- **vllm-profiler assumes K8s** — the webhook-based injection requires a K8s cluster. Can we use the standalone `sitecustomize.py` approach without K8s?
- **Profiler trace storage** — where do traces go? S3 bucket (as AI-Analysis-Agent expects) or local filesystem?
- **Which vLLM version(s) to profile?** The kernel-to-code mapper has curated mappings — do these cover the version we'll use?

### Dashboard

- **Which dashboard do we serve?** performance-dashboard has 3 dashboards (RHAIIS, LLM-D, MLPerf). Do we need all three, or just RHAIIS?
- **IntelliConfig wizard** — is the config recommendation wizard part of the deliverable, or just the benchmarking pipeline?

### Integration

- **How do the 5 repos connect?** The repos were built independently. The integration path is unclear:
  - vllm-profiler generates traces -> AI-Analysis-Agent analyzes them
  - GuideLLM runs benchmarks -> results go to performance-dashboard
  - Agent loop ties it together — but which agent (Gemini-based or Claude-based)?
- **MCP server reuse** — AI-Analysis-Agent's 28 MCP tools are the richest component. Can the agent call these tools directly, or do we need to adapt them?
- **Data flow** — what is the end-to-end pipeline? Deploy vLLM -> profile -> benchmark -> analyze -> recommend -> re-configure -> re-benchmark -> report?

### Deliverables

- **What does "automated evaluation pipeline" mean concretely?** A script that sweeps configs? A dashboard? An agent that converses with the user?
- **"Catch regressions and surface bottlenecks"** — is this comparing across vLLM versions, or across config variations within a single version?
- **Demo format** — live demo on a cluster, or a recorded walkthrough with pre-collected data?

---

## Quick Reference: Key Files by Function

| Need | Go to |
|------|-------|
| Agentic loop pattern | `ai-perf-hackathon/agent/agentic.py` |
| MCP tool server (28 tools) | `AI-Analysis-Agent/psap-mcp-server/` |
| LangGraph agent setup | `AI-Analysis-Agent/psap-agent/psap_agent/src/core/agent.py` |
| Benchmark methodology | `LLM-inference-benchmark-guide/README.md` |
| Performance dashboard | `performance-dashboard/dashboard.py` |
| PyTorch profiler injection | `vllm-profiler/sitecustomize.py` |
| Kernel-to-code mapping | `AI-Analysis-Agent/psap-mcp-server/.../tools/kernel_code_mapper_tool.py` |
| Grafana metric queries | `AI-Analysis-Agent/psap-mcp-server/.../tools/grafana_metrics_tool.py` |
| Cost analysis | `AI-Analysis-Agent/psap-mcp-server/.../tools/cost_efficiency_tool.py` |
| Config generator (IntelliConfig) | `performance-dashboard/intelliconfig.py` |
| vLLM runtime params reference | `LLM-inference-benchmark-guide/README.md` |
| Token/cost tracking | `ai-perf-hackathon/agent/llm.py` |
| Safe remediation pattern | `ai-perf-hackathon/agent/remediator.py` |
| K8s profiler deployment | `vllm-profiler/deploy.sh` |
