# Policy-expressibility paper corpus

This directory is deliberately separate from `../ref-paper/`. None of the four
known-mismatched files in that directory were read, overwritten, or reused.

As of 2026-09-02, the catalog contains 59 primary-source entries. Fifty-four
PDFs (96,870,994 bytes) were retained. Every retained file passed `pdfinfo` and
page-one `pdftotext` extraction, followed by a manual title-and-author check.
Five entries could not be retained: one is source code rather than a paper, two
ACM endpoints rejected automated retrieval, one advertised author link is
missing, and one former author host no longer has a DNS address. No hashes,
checksums, digests, or fingerprints were generated.

`MANIFEST.json` is the canonical machine-readable inventory. The table below
contains the same required provenance and validation fields in compact form.

| ID | Title (year, venue) | Source landing page | Direct PDF | Local file / bytes | Title check |
|---:|---|---|---|---|---|
| 1 | NVIDIA Unified Virtual Memory native implementation (2026, source) | [NVIDIA repository](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/main/kernel-open/nvidia-uvm) | — | — | `no-public-pdf` (not a paper) |
| 2 | Towards High Performance Paged Memory for GPUs (2016, HPCA) | [NVIDIA Research](https://research.nvidia.com/publication/2016-03_towards-high-performance-paged-memory-gpus) | [PDF](https://www.cs.utexas.edu/~skeckler/pubs/HPCA_2016_Paged_Memory.pdf) | `02-hpca16-paged-memory.pdf` / 2,277,479 | verified title and authors |
| 3 | Interplay between Hardware Prefetcher and Page Eviction Policy in CPU-GPU Unified Virtual Memory (2019, ISCA) | [DOI](https://doi.org/10.1145/3307650.3322224) | [PDF](https://people.cs.pitt.edu/~debashis/papers/ISCA2019.pdf) | `03-isca19-prefetch-eviction.pdf` / 3,110,203 | verified title and authors |
| 4 | Adaptive Page Migration for Irregular Data-Intensive Applications under GPU Memory Oversubscription (2020, IPDPS) | [DOI](https://doi.org/10.1109/IPDPS47924.2020.00054) | [PDF](https://people.cs.pitt.edu/~debashis/papers/IPDPS2020.pdf) | `04-ipdps20-adaptive-page-migration.pdf` / 632,227 | verified title and authors |
| 5 | An Adaptive Framework for Oversubscription Management in CPU-GPU Unified Memory (2021, DATE) | [DOI](https://doi.org/10.23919/DATE51398.2021.9473982) | [PDF](https://past.date-conference.com/proceedings-archive/2021/pdf/1974.pdf) | `05-date21-adaptive-oversubscription.pdf` / 1,461,577 | verified title and authors |
| 6 | A Framework for Memory Oversubscription Management in Graphics Processing Units (2019, ASPLOS) | [DOI](https://doi.org/10.1145/3297858.3304044) | [PDF](https://rausavar.github.io/pubs/li_asplos19_final.pdf) | `06-asplos19-etc.pdf` / 2,198,515 | verified title and authors |
| 7 | Batch-Aware Unified Memory Management in GPUs for Irregular Workloads (2020, ASPLOS) | [DOI](https://doi.org/10.1145/3373376.3378529) | [PDF](https://ramyadhadidi.github.io/files/kim-asplos20.pdf) | `07-asplos20-batch-aware-uvm.pdf` / 992,272 | verified title and authors |
| 8 | DeepUM: Tensor Migration and Prefetching in Unified Memory (2023, ASPLOS) | [DOI](https://doi.org/10.1145/3575693.3575736) | [PDF](https://dl.acm.org/doi/pdf/10.1145/3575693.3575736) | — | `download-failed-http-403` |
| 9 | HELM: Characterizing Unified Memory Accesses to Improve GPU Performance under Memory Oversubscription (2025, SC) | [DOI](https://doi.org/10.1145/3712285.3759812) | [PDF](https://dl.acm.org/doi/pdf/10.1145/3712285.3759812) | — | `download-failed-http-403` |
| 10 | An Intelligent Framework for Oversubscription Management in CPU-GPU Unified Memory (2022, arXiv) | [arXiv](https://arxiv.org/abs/2204.02974) | [PDF](https://arxiv.org/pdf/2204.02974) | `10-2022-intelligent-oversubscription.pdf` / 543,585 | verified title and authors |
| 11 | Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling (2025, arXiv) | [arXiv](https://arxiv.org/abs/2512.24637) | [PDF](https://arxiv.org/pdf/2512.24637) | `11-2025-msched.pdf` / 523,276 | verified title and authors |
| 12 | Page Placement Strategies for GPUs within Heterogeneous Memory Systems (2015, ASPLOS) | [NVIDIA Research](https://research.nvidia.com/publication/2015-03_page-placement-strategies-gpus-within-heterogeneous-memory-systems) | [PDF](https://research.nvidia.com/sites/default/files/pubs/2015-03_Page-Placement-Strategies//agarwal.asplos2015.pdf) | `12-asplos15-page-placement.pdf` / 2,501,811 | verified title and authors |
| 13 | Mosaic: A GPU Memory Manager with Application-Transparent Support for Multiple Page Sizes (2017, MICRO) | [DOI](https://doi.org/10.1145/3123939.3123975) | [PDF](https://ghose.cs.illinois.edu/papers/17micro_mosaic.pdf) | `13-micro17-mosaic.pdf` / 1,736,361 | verified title and authors |
| 14 | G10: Enabling An Efficient Unified GPU Memory and Storage Architecture with Smart Tensor Migrations (2023, MICRO) | [official repository](https://github.com/platformxlab/G10) | [PDF](https://jianh.web.engr.illinois.edu/papers/g10-micro23.pdf) | `14-micro23-g10.pdf` / 4,925,907 | verified title and authors |
| 15 | DREAM: Device-Driven Efficient Access to Virtual Memory (2025, ICS) | [official repository](https://github.com/nnurlan008/dream) | [PDF](https://www.cs.ucr.edu/~elaheh/papers/ICS2025-Nurlan.pdf) | `15-ics25-dream.pdf` / 802,545 | verified title and authors |
| 16 | SUV: Static Analysis Guided Unified Virtual Memory (2024, MICRO) | [DOI](https://doi.org/10.1109/MICRO61859.2024.00030) | [PDF](https://guilhermecox.github.io/dw/pratheek-micro24.pdf) | `16-micro24-suv.pdf` / 582,927 | verified title and authors |
| 17 | Forest: Access-aware GPU UVM Management (2025, ISCA) | [DOI](https://doi.org/10.1145/3695053.3731047) | [advertised author PDF](https://guilhermecox.github.io/dw/lin-isca25.pdf) | — | `download-failed-author-link-404-and-publisher-403` |
| 18 | OASIS: Object-Aware Page Management for Multi-GPU Systems (2025, HPCA) | [DOI](https://doi.org/10.1109/HPCA61900.2025.00124) | [PDF](https://yueqiwang42.github.io/assets/pdf/papers/OASIS_HPCA25.pdf) | `18-hpca25-oasis.pdf` / 847,525 | verified title and authors |
| 19 | GRIT: Enhancing Multi-GPU Performance with Fine-Grained Dynamic Page Placement (2024, HPCA) | [DOI](https://doi.org/10.1109/HPCA57654.2024.00085) | [PDF](https://yueqiwang42.github.io/assets/pdf/papers/GRIT_HPCA24.pdf) | `19-hpca24-grit.pdf` / 479,202 | verified title and authors |
| 20 | MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache (2024, arXiv) | [arXiv](https://arxiv.org/abs/2401.14361) | [PDF](https://arxiv.org/pdf/2401.14361) | `20-2024-moe-infinity.pdf` / 589,193 | verified title and authors |
| 21 | Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models (2025, ICLR) | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8cd1ce03ea58b3d7dfd809e4d42f08ea-Abstract-Conference.html) | [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf) | `21-iclr25-fiddler.pdf` / 2,528,059 | verified title and authors |
| 22 | HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference (2024, arXiv) | [arXiv](https://arxiv.org/abs/2411.01433) | [PDF](https://arxiv.org/pdf/2411.01433) | `22-2024-hobbit.pdf` / 5,401,977 | verified title and authors |
| 23 | MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs (2025, ASPLOS) | [DOI](https://doi.org/10.1145/3669940.3707267) | [PDF](https://pschafhalter.com/papers/2025-asplos-moe-lightning.pdf) | `23-asplos25-moe-lightning.pdf` / 4,967,923 | verified title and authors |
| 24 | ProMoE: Fast MoE-based LLM Serving using Proactive Caching (2024, arXiv) | [arXiv](https://arxiv.org/abs/2410.22134) | [PDF](https://arxiv.org/pdf/2410.22134) | `24-2024-promoe.pdf` / 1,336,214 | verified title and authors |
| 25 | Accelerating Distributed MoE Training and Inference with Lina (2023, USENIX ATC) | [USENIX](https://www.usenix.org/conference/atc23/presentation/li-jiamin) | [PDF](https://www.usenix.org/system/files/atc23-li-jiamin.pdf) | `25-atc23-lina.pdf` / 3,563,229 | verified title and authors |
| 26 | PopFetcher: Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch (2025, USENIX ATC) | [USENIX](https://www.usenix.org/conference/atc25/presentation/zhang-junyi) | [PDF](https://www.usenix.org/system/files/atc25-zhang-junyi.pdf) | `26-atc25-popfetcher.pdf` / 4,392,401 | verified title and authors |
| 27 | DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference (2025, arXiv) | [arXiv](https://arxiv.org/abs/2501.10375) | [PDF](https://arxiv.org/pdf/2501.10375) | `27-2025-daop.pdf` / 482,866 | verified title and authors |
| 28 | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU (2023, ICML) | [PMLR](https://proceedings.mlr.press/v202/sheng23a.html) | [PDF](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) | `28-icml23-flexgen.pdf` / 524,133 | verified title and authors |
| 29 | InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management (2024, OSDI) | [USENIX](https://www.usenix.org/conference/osdi24/presentation/lee) | [PDF](https://www.usenix.org/system/files/osdi24-lee.pdf) | `29-osdi24-infinigen.pdf` / 3,212,724 | verified title and authors |
| 30 | Efficient Memory Management for Large Language Model Serving with PagedAttention (2023, SOSP) | [DOI](https://doi.org/10.1145/3600006.3613165) | [PDF](https://arxiv.org/pdf/2309.06180) | `30-sosp23-pagedattention-vllm.pdf` / 1,459,631 | verified title and authors |
| 31 | PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU (2024, SOSP) | [DOI](https://doi.org/10.1145/3694715.3695964) | [PDF](https://ipads.se.sjtu.edu.cn/_media/publications/song-sosp24.pdf) | `31-sosp24-powerinfer.pdf` / 986,808 | verified title and authors |
| 32 | ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning (2021, SC) | [arXiv](https://arxiv.org/abs/2104.07857) | [PDF](https://arxiv.org/pdf/2104.07857) | `32-sc21-zero-infinity.pdf` / 1,322,978 | verified title and authors |
| 33 | DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale (2022, SC) | [Microsoft Research](https://www.microsoft.com/en-us/research/project/deepspeed/publications/) | [PDF](https://raw.githubusercontent.com/microsoft/DeepSpeed/master/docs/assets/files/sc22-ds-inference.pdf) | `33-sc22-deepspeed-inference.pdf` / 1,842,235 | verified title and authors |
| 34 | Capuchin: Tensor-based GPU Memory Management for Deep Learning (2020, ASPLOS) | [DOI](https://doi.org/10.1145/3373376.3378505) | [former author PDF](https://alchem.usc.edu/portal/static/download/capuchin.pdf) | — | `download-failed-author-host-no-dns` |
| 35 | ServerlessLLM: Low-Latency Serverless Inference for Large Language Models (2024, OSDI) | [USENIX](https://www.usenix.org/conference/osdi24/presentation/fu) | [PDF](https://www.usenix.org/system/files/osdi24-fu.pdf) | `35-osdi24-serverlessllm.pdf` / 968,758 | verified title and authors |
| 36 | Serving DNNs like Clockwork: Performance Predictability from the Bottom Up (2020, OSDI) | [USENIX](https://www.usenix.org/conference/osdi20/presentation/gujarati) | [PDF](https://www.usenix.org/system/files/osdi20-gujarati.pdf) | `36-osdi20-clockwork.pdf` / 856,804 | verified title and authors |
| 37 | Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences (2022, OSDI) | [USENIX](https://www.usenix.org/conference/osdi22/presentation/han) | [PDF](https://www.usenix.org/system/files/osdi22-han.pdf) | `37-osdi22-reef.pdf` / 804,508 | verified title and authors |
| 38 | Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications (2020, MLSys) | [MLSys](https://proceedings.mlsys.org/paper_files/paper/2020/hash/d9cd83bc91b8c36a0c7c0fcca59228f2-Abstract.html) | [PDF](https://proceedings.mlsys.org/paper_files/paper/2020/file/d9cd83bc91b8c36a0c7c0fcca59228f2-Paper.pdf) | `38-mlsys20-salus.pdf` / 1,392,423 | verified title and authors |
| 39 | Gandiva: Introspective Cluster Scheduling for Deep Learning (2018, OSDI) | [USENIX](https://www.usenix.org/conference/osdi18/presentation/xiao) | [PDF](https://www.usenix.org/system/files/osdi18-xiao.pdf) | `39-osdi18-gandiva.pdf` / 1,170,496 | verified title and authors |
| 40 | PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications (2020, OSDI) | [USENIX](https://www.usenix.org/conference/osdi20/presentation/bai) | [PDF](https://www.usenix.org/system/files/osdi20-bai.pdf) | `40-osdi20-pipeswitch.pdf` / 929,126 | verified title and authors |
| 41 | Transparent GPU Sharing in Container Clouds for Deep Learning Workloads (2023, NSDI) | [USENIX](https://www.usenix.org/conference/nsdi23/presentation/wu) | [PDF](https://www.usenix.org/system/files/nsdi23-wu.pdf) | `41-nsdi23-tgs.pdf` / 1,498,652 | verified title and authors |
| 42 | PREMA: A Predictive Multi-task Scheduling Algorithm For Preemptible Neural Processing Units (2020, HPCA) | [DOI](https://doi.org/10.1109/HPCA47549.2020.00027) | [PDF](https://arxiv.org/pdf/1909.04548) | `42-hpca20-prema.pdf` / 5,003,893 | verified title and authors |
| 43 | Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing (2022, USENIX ATC) | [USENIX](https://www.usenix.org/conference/atc22/presentation/choi-seungbeom) | [PDF](https://www.usenix.org/system/files/atc22-choi-seungbeom.pdf) | `43-atc22-gpulets.pdf` / 7,567,937 | verified title and authors |
| 44 | Zico: Efficient GPU Memory Sharing for Concurrent DNN Training (2021, USENIX ATC) | [USENIX](https://www.usenix.org/conference/atc21/presentation/lim) | [PDF](https://www.usenix.org/system/files/atc21-lim.pdf) | `44-atc21-zico.pdf` / 2,410,694 | verified title and authors |
| 45 | Memory Harvesting in Multi-GPU Systems with Hierarchical Unified Virtual Memory (2022, USENIX ATC) | [USENIX](https://www.usenix.org/conference/atc22/presentation/choi-sangjin) | [PDF](https://www.usenix.org/system/files/atc22-choi-sangjin_1.pdf) | `45-atc22-huvm.pdf` / 1,545,111 | verified title and authors |
| 46 | GPS: A Global Publish-Subscribe Model for Multi-GPU Memory Management (2021, MICRO) | [NVIDIA Research](https://research.nvidia.com/publication/2021-10_gps-global-publish-subscribe-model-multi-gpu-memory-management) | [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2021_PublishSubscribe.pdf) | `46-micro21-gps.pdf` / 1,812,654 | verified title and authors |
| 47 | Combining HW/SW Mechanisms to Improve NUMA Performance of Multi-GPU Systems (2018, MICRO) | [NVIDIA Research](https://research.nvidia.com/publication/2018-10_combining-hwsw-mechanisms-improve-numa-performance-multi-gpu-systems) | [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2018_CARVE.pdf) | `47-micro18-carve.pdf` / 863,202 | verified title and authors |
| 48 | Griffin: Hardware-Software Support for Efficient Page Migration in Multi-GPU Systems (2020, HPCA) | [DOI](https://doi.org/10.1109/HPCA47549.2020.00055) | [PDF](https://sarchlab.org/hpca2020.pdf) | `48-hpca20-griffin.pdf` / 408,470 | verified title and authors |
| 49 | Beyond the Socket: NUMA-Aware GPUs (2017, MICRO) | [NVIDIA Research](https://research.nvidia.com/publication/2017-10_Beyond-the-socket%3A) | [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2017_TMG.pdf) | `49-micro17-beyond-the-socket.pdf` / 1,264,331 | verified title and authors |
| 50 | Extending Applications Safely and Efficiently / bpftime (2025, OSDI) | [USENIX](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) | [PDF](https://www.usenix.org/system/files/osdi25-zheng-yusheng.pdf) | `50-osdi25-bpftime-eim.pdf` / 691,321 | verified title and authors |
| 51 | eGPU: Extending eBPF Programmability and Observability to GPUs (2025, HCDS) | [Publisher](https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/HCDS25/10/13a8f7c0-0a7e-11f0-ada9-16bb50361d1f/OUT/hcds25-10.html) | [PDF](https://asplos.dev/pdf/bpftime_super.pdf) | `51-hcds25-egpu.pdf` / 609,048 | verified title and authors |
| 52 | Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications (2024, EuroSys) | [DOI](https://doi.org/10.1145/3627703.3629578) | [PDF](https://fotstrt.github.io/files/2024-orion.pdf) | `52-eurosys24-orion.pdf` / 1,894,950 | verified title and authors |
| 53 | Paella: Low-latency Model Serving with Software-defined GPU Scheduling (2023, SOSP) | [DOI](https://doi.org/10.1145/3600006.3613163) | [PDF](https://vincen.tl/files/ng23paella.pdf) | `53-sosp23-paella.pdf` / 1,203,901 | verified title and authors |
| 54 | Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads (2025, ASPLOS) | [DOI](https://doi.org/10.1145/3669940.3707282) | [PDF](https://arxiv.org/pdf/2410.07381) | `54-asplos25-tally.pdf` / 1,176,509 | verified title and authors |
| 55 | LithOS: An Operating System for Efficient Machine Learning on GPUs (2025, SOSP) | [arXiv](https://arxiv.org/abs/2504.15465) | [PDF](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/lithos_sosp25.pdf) | `55-sosp25-lithos.pdf` / 1,693,307 | verified title and authors |
| 56 | Kernelet: High-Throughput GPU Kernel Executions with Dynamic Slicing and Scheduling (2013, arXiv manuscript) | [arXiv](https://arxiv.org/abs/1303.5164) | [PDF](https://arxiv.org/pdf/1303.5164) | `56-2013-kernelet.pdf` / 1,039,256 | verified title and authors |
| 57 | Improving GPU Sharing Performance through Adaptive Bubbleless Spatial-Temporal Sharing / Bless (2025, EuroSys) | [DOI](https://doi.org/10.1145/3689031.3696070) | [PDF](https://jamesthez.github.io/files/bless-eurosys25.pdf) | `57-eurosys25-bless.pdf` / 2,327,832 | verified title and authors |
| 58 | TimeGraph: GPU Scheduling for Real-Time Multi-Tasking Environments (2011, USENIX ATC) | [USENIX](https://www.usenix.org/conference/usenixatc11/timegraph-gpu-scheduling-real-time-multi-tasking-environments) | [PDF](https://www.usenix.org/legacy/events/atc11/tech/final_files/Kato.pdf) | `58-atc11-timegraph.pdf` / 493,655 | verified title and authors |
| 59 | Gdev: First-Class GPU Resource Management in the Operating System (2012, USENIX ATC) | [USENIX](https://www.usenix.org/system/files/conference/atc12/atc12-final319.pdf) | [PDF](https://www.usenix.org/system/files/conference/atc12/atc12-final319.pdf) | `59-atc12-gdev.pdf` / 1,018,373 | verified title and authors |

## Verification commands

Run these from this directory. They validate PDF structure and extract the first
page for human title-and-author inspection; they do not calculate hashes.

```bash
python3 validate_manifest.py

find . -maxdepth 1 -type f -name '*.pdf' -print0 \
  | xargs -0 -n1 pdfinfo

for paper in ./*.pdf; do
  pdftotext -f 1 -l 1 -layout "$paper" -
done

jq empty MANIFEST.json
```
