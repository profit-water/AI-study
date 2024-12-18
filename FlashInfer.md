# FlashInfer Overview
  **FlashInfer**는 LLM(Large Language Model) 서빙을 가속화하기 위해 개발된 오픈소스 라이브러리.(Apache 2.0 License).  
  University of Washington, Carnegie Mellon University, OctoAI 연구자들이 2023년 여름부터 공동 개발.  
  PyTorch API를 통해 빠른 프로토타이핑 지원(import flashinfer 로 사용가능), 헤더 전용 C++ API로 LLM 서빙 시스템에 쉽게 통합 가능.  
  다양한 LLM 서빙 시나리오에서 최첨단 성능 제공.

---

## Key Features
  1. **Comprehensive Attention Kernels**
     - Padded Tensor, Ragged Tensor, Page Table 등 다양한 KV-Cache 포맷 지원
     - Single-request와 Batching 시나리오를 위한 Prefill, Decode, Append 커널 최적화 제공

  2. **Optimized Shared-Prefix Batch Decoding**
     - Cascading 기법을 활용해 Shared-Prefix Decoding 성능 개선
     - Long Prompt(32,768 tokens)와 Large Batch Size(256) 기준 최대 **31x 속도 향상** 달성

  3. **Accelerated Attention for Compressed/Quantized KV-Cache**
     - Grouped-Query Attention, Fused-RoPE Attention, Quantized Attention 최적화
     - A100 및 H100 GPU 기준 **vllm 대비 2-3x 속도 향상** 제공

---

## Adoption and Use Cases
  FlashInfer는 다음 시스템들에 사용 중:
  - **MLC-LLM** (CUDA backend)
  - **Punica**
  - **sglang**
  - **sarathi-serve**
  - **vllm**
  - **TGI3.0**

[FlashInfer GitHub Repository](https://github.com/flashinfer-ai/flashinfer).

---

## FlashInfer 에 관련된 Attention 상식(?)

### Attentions in LLM Serving
  LLM 서빙은 Prefill, Decode, Append 세 단계로 나뉨. 단계별 KV-Cache와 Query 간 Attention 계산 방식이 다름.

  1. **Prefill Stage**
     - KV-Cache와 모든 Query 간 Attention 계산 수행
     - Causal Mask 하에서 전체 Attention 맵 채워짐
     - 연산 강도가 높아 GPU 계산 성능(Compute-Bound)에 제한됨
     - ![image](https://github.com/user-attachments/assets/0eb59bd0-b39e-4481-9f97-c75e1409bcc2)

  2. **Decode Stage**
     - 모델이 한 번에 한 개의 토큰 생성, KV-Cache와 단일 Query 간 Attention 계산
     - Attention 맵의 한 행씩 채워짐
     - Query 길이가 짧아 GPU 메모리 대역폭(I/O-Bound)에 제한됨
     - ![image](https://github.com/user-attachments/assets/b0353c5c-060f-4c3e-96e2-83084d492969)

  3. **Append Stage**
     - KV-Cache와 추가된 Query 토큰 간 Attention 계산
     - Speculative Decoding(추론적 디코딩)에 사용
     - **참조** speculative decoding이란?
     - ![image](https://github.com/user-attachments/assets/7a36719e-81d0-400e-9c1a-10ec85abd889)
     - Query 길이에 따라 IO-Bound에서 Compute-Bound로 전환 가능
     - ![image](https://github.com/user-attachments/assets/f439839c-59a4-43d4-a676-ad5e32f1a71b)

---

### Key Factors in Attention Efficiency
  Attention 계산 효율성은 Query 길이(Q)에 좌우됨. 연산 강도(Operational Intensity)는 \( Q \cdot K \)로 정의되며, \( K \)는 KV-Cache 길이.

  - Decode Stage: \( Q = 1 \)일 때 연산 강도가 낮아 IO-Bound 상태
  - Prefill Stage: \( Q \cdot K^T \) 값이 커지면서 Compute-Bound 상태
  - Append Stage: Query 길이에 따라 IO-Bound와 Compute-Bound 전환

---

### Roofline Model of Attention Operators
  - Decode Attention: GPU 메모리 대역폭에 제한, 항상 IO-Bound 상태
  - Prefill Attention: 높은 연산 강도로 GPU 계산 성능에 제한, Compute-Bound 상태
  - Append Attention: Query 길이에 따라 IO-Bound에서 Compute-Bound로 전환
  - ![image](https://github.com/user-attachments/assets/4ec51649-3ac4-48cc-b056-abae5be95fd2)

---

# FlashInfer 주요기능

1. **FlashAttention 및 FlashAttention2 통합**
   - Multi-head attention을 단일 커널로 통합하여 GPU global memory로의 Attention Matrix 저장 오버헤드 제거.(FlashAttention 1)
   - ![image](https://github.com/user-attachments/assets/e8322861-a4ee-4ec2-8bdf-96dd7ff1928b)
   - FlashAttention2는 타일링 전략 개선 및 non-tensor 연산 감소로 A100/H100 GPU의 낮은 non-tensor core 성능 문제를 완화.
   - 기존의 scailing 공식을 조금 변형하여, non-tensor 연산을 감소시킴, inner loop가 K로 바뀜
   - ![image](https://github.com/user-attachments/assets/2aa61f5c-d5c6-42c9-8b9a-e12ad27a82e2)

2. **PageAttention 지원**
   - KV-Cache를 페이지 테이블로 구성하여 Memory fragmentation 문제 해결.

3. **Prefill, Decode, Append 커널 구현**
   - 단일 요청 및 배치 버전에 대해 FlashAttention을 구현하여 모든 KV-Cache 포맷(Ragged Tensor, Page Table 등) 지원.
   - 기존 라이브러리에서 구현하지 못한 Paged KV-Cache를 위한 Prefill/Append 커널 구현, Speculative Decoding 환경에서 사용 가능.

4. **KV-Cache 압축을 위한 커널 최적화**
   - **Grouped Query Attention (GQA)**  
     - Key와 Value에 대해 적은 수의 Head를 사용하여 메모리 트래픽 감소.  
     - 기존 GQA는 GPU의 낮은 non-tensor core 성능으로 인해 Compute-Bound 상태가 됨.  
     - FlashInfer는 Tensor Core를 활용한 Prefill 커널을 사용하여 GQA 디코드 Attention에서 최대 **2-3x 속도 향상** 달성.  

   - **Fused-RoPE Attention**  
     - RoPE(Rotary Positional Embeddings) 적용을 KV-Cache 내부에서 통합.  
     - 토큰 위치 변경 후에도 Pre-RoPE Key 저장 및 Attention 커널 내 RoPE 적용으로 효율성 유지.  
     - 다양한 플랫폼에서 **미미한 오버헤드로 RoPE 적용** 가능.  

   - **Quantized Attention**  
     - KV-Cache를 4비트까지 압축(예: FlexGen, Atom)하여 정확도 손실 없이 메모리 사용량 감소.  
     - FlashInfer는 저정밀도 Attention 커널을 구현하여 압축 비율에 비례한 속도 향상(예: 4비트에서 **4배**, 8비트에서 **2배** 속도 향상).  

5. **PageAttention 최적화**
   - 페이지 크기가 1인 특수 PageAttention 환경에서도 GPU 공유 메모리를 활용한 page index pre-fetching 으로 성능 저하 방지.
     ( 자세하게는 이해 못함 )

---

# What's new in FlashInfer?

1. Cascade Inference : Memory Bandwidth Efficient Shared Prefix Batch Decoding
  - LLM 추론 작업은 공유 프리픽스(프롬프트)에서 여러 독립적인 출력을 생성하는 경우가 많음.
  - 대표적인 예로 Self-Consistency, Tree of Thoughts, Skeleton-of-Thought 기술이 있음. 
  - 이러한 prefix가 길고 요청 수가 많을 경우 메모리 및 시간 소모가 큼. 
  - 예를 들어, **긴 문서 기반 QA**에서는 여러 사용자가 동일한 문서를 프롬프트로 사용하여 챗봇과 상호작용하는 경우가 이에 해당.
  - ![image](https://github.com/user-attachments/assets/7acf7832-315d-4ba7-9ddf-507956dd0fea)

---

## Background - 기존의 문제점

1. **메모리 사용량**  
   - KV-Cache에 shared prefix를 여러번 저장하면 메모리 사용량이 크게 증가.  
   - vLLM 은 shared prefix를 한 번만 저장하여 메모리 문제를 완화하지만, 여전히 효율적인 데이터 접근은 어려움.

2. **성능 병목**  
   - vLLM의 기본 PageAttention 구현은 공유 프롬프트에 대한 KV-Cache 접근을 최적화하지 않아 시간 효율성이 낮음.

3. 참고 - GPU 메모리 계층 구조

   - **메모리 구성요소**:
     - **글로벌 메모리 & L2 캐시**: 모든 SM(Streaming Multiprocessor) 간 공유.  
     - **공유 메모리(SMEM) & 레지스터**: 각 SM에만 국한되며 훨씬 높은 처리량 제공.

   - **SMEM의 중요성**:
     - 글로벌 메모리나 L2 캐시 접근보다 SMEM 및 레지스터 접근이 훨씬 빠름.  
     - 글로벌 메모리 접근을 최소화하고 SMEM 사용을 극대화해야 GPU 처리량 향상 가능.

   - ![image](https://github.com/user-attachments/assets/155f96c3-bc07-465c-8e21-1e3884471c03)

---

## Cascade Inference: 해결책

  **Cascade Inference**는 공유 prefix와 고유 suffix에 대한 Attention 계산을 분리하여 공유  
    KV-Cache를 GPU의 공유 메모리(SMEM)에 저장해 빠르게 접근 가능하도록 개선. 

  - 주요 이점:

  1. **효율성**:  
    - SMEM은 global memory(HBM) 이나 L2 캐시보다 높은 처리량을 제공하므로 메모리 접근 병목을 최소화.  
    - H100 SXM 80GB GPU 기준, vLLM PageAttention 대비 **31배**, Cascade 미적용 FlashInfer 배치 디코딩 대비 **26배 속도 향상**.

  2. **구현**:  
    - FlashInfer에서 PyTorch 및 C++ API를 통해 지원.  
    - shared prefix batch decoding 연산자에 최적화.

---

## Cascade Inference movivation ->  Multi-Query vs. Single-Query CUDA 커널 차이
  ![image](https://github.com/user-attachments/assets/cb20bf12-a3ed-4696-aed1-19b5e6cb83af)

  ### Multi-Query Attention 커널 (Prefill/Append)
  - 특징:
    - 여러 Query가 동일한 KV-Cache 영역에 접근.
    - KV-Cache를 SMEM에 로드하고 단일 Thread Block에서 병렬로 처리.
  - 장점:
    - KV-Cache 재사용으로 대역폭 효율성 증가.
    - Tensor Core 활용으로 최대 TFLOPs/s 달성.
  - 제한:
    - Query마다 KV-Cache 영역이 다를 경우 적용 불가능.

  ### Single-Query Attention 커널 (Decode)
  - 특징:
    - 각 Query가 고유한 KV-Cache를 가짐.
    - 병렬 처리를 유지하기 위해 Query당 하나의 Thread Block을 할당.
  - 단점:
    - Query 간 KV-Cache 재사용 기회가 없어 비효율적.
    - 각 Thread Block이 글로벌 메모리 또는 L2 캐시에서 KV-Cache를 로드해야 하므로 메모리 대역폭 효율성 낮음.

---

## Divide and Conquer : Shared-prefix batch decoding 최적화
  Shared-prefix batch decoding을 최적화하려면 두 커널의 강점을 결합해야 함:
  - **Multi-Query Attention**으로 공유 프리픽스를 처리.
  - **Single-Query Attention**으로 고유 서픽스를 처리.
  - 두 계산 결과를 병합하여 최적의 성능을 달성.

## How to merge?
  attention 에 대한 merge 연산자를 새롭게 정의
  ![image](https://github.com/user-attachments/assets/f332b1a1-f862-4ef0-9c70-9bf4c106274f)
  - 해당 연산자는 결합/교환 법칙 모두 성립함, 따라서 아래와 같은 연산 결합이 가능함.
  - ![image](https://github.com/user-attachments/assets/6525d620-0bb2-49da-afab-6c2516204028)

## Cascade Inference : The Algorithm
  **Recursive Attention**은 Attention 계산을 여러 단계로 분해하고,  
  각 단계를 서로 다른 연산 장치 또는 컴퓨팅 유닛에 할당할 수 있도록 지원함.  
  FlashInfer와 Flash-Decoding에서 사용되는 KV 시퀀스 분할(trick)은 동일한 아이디어를 활용하여 서로 다른 Thread Block에서  
  생성된 부분 Attention 상태를 병합하는 방식임.

  ### 알고리즘 단계
  1. **Multi-Query Attention Kernel 사용**:
     - Query와 공유 프리픽스의 KV-Cache 간 Attention 상태를 계산.
     - Prefill/Append 단계에서 사용되며, SMEM(공유 메모리) 또는 레지스터를 통해 KV-Cache에 접근하여 효율성을 극대화.

  2. **Batch Decode Attention Kernel 사용**:
     - Query와 고유 서픽스의 KV-Cache 간 Attention 상태를 계산.
     - 디코드 단계에서 사용되며, L2 Cache 또는 글로벌 메모리를 통해 KV-Cache에 접근.

  3. **Merge Operator 활용**:
     - 두 Attention 상태를 병합하여 최종 Attention 출력을 생성.
   
  ![image](https://github.com/user-attachments/assets/84c93118-fc47-4e46-9935-bfc303160c81)

  - 다른 색상의 직사각형은 GPU의 서로 다른 Thread Block에서 처리됨.
  - Multi-Query Attention Kernel은 공유 prefix를 SMEM/레지스터로 처리.
  - Batch Decode Kernel은 고유 suffix를 L2 Cache/글로벌 메모리로 처리.
  - 최종적으로 Merge Operator를 사용하여 두 계산 결과를 병합.

---

## Evaluations : Cascade Inference 성능 평가
  Cascade Inference는 H100 SXM 80GB와 A100 PCIE 80GB GPU에서 평가됨.  
  입력 형태는 LLaMA2-7B(32 heads, d_hidden_per_head = 128)에서 가져옴.  
  세 가지 매개변수를 변경하며 실험 진행:
  1. 요청 수 (batch size)
  2. shared prefix 길이
  3. 요청별 unique suffix 길이

  ![image](https://github.com/user-attachments/assets/03f31c1d-a0bf-4cc2-b791-067dff86fe89)

  ### 결과 요약
  - FlashInfer 커널 (Cascading 적용 및 비적용):
    - vLLM 커널보다 항상 높은 성능을 보임.
    - Cascading 커널은 비-Cascading 커널 대비 대부분의 경우에서 상당한 속도 향상.

  - Cascade Inference의 이점:
    - 공유 프리픽스 길이 및 배치 크기 증가 시 이점 증가:
      - Prefill 커널이 실행 시간에서 지배적인 비중을 차지할 때 성능 향상 두드러짐.
    - 고유 서픽스 길이 증가 시 이점 감소:
      - Batch Decode 커널이 실행 시간에서 더 큰 비중을 차지.

  - 긴 공유 프롬프트 (32768) 실험:
    - H100 SXM 80GB, 큰 배치 크기(≥128) 및 짧은 고유 KV 길이(≤256) 조건에서 디코드 커널이 최대 31배 속도 향상.

2. KV-Cache layout in FlashInfer
   FlashInfer는 KV-Cache의 마지막 3차원에 대해 두 가지 레이아웃을 제공:
   1. **NHD (seq_len, num_heads, head_dim)**
   2. **HND (num_heads, seq_len, head_dim)**

---

## NHD Layout
  - 구조: `(seq_len, num_heads, head_dim)`
  - 장점:
    - Attention 연산 출력과 일관성을 유지하므로 Transpose 없이 바로 사용 가능.
    - 가독성이 더 좋음.
  - 사용 사례: FP16 기반 KV-Cache에서 주로 사용되며, 기본값으로 설정됨.

---

## HND Layout
  - 구조: `(num_heads, seq_len, head_dim)`
  - 장점:
    - GPU 구현에 유리, 특히 FP8과 같은 저정밀도 데이터 타입을 사용하는 경우 효율적.(why???)
  - 사용 사례: 저정밀도 데이터 타입(FP8 등)에서 성능 최적화.

---

## 성능 비교
  - FP16 기반 KV-Cache에서는 NHD와 HND 레이아웃 간의 성능 차이가 미미함.
  - FlashInfer는 두 레이아웃 모두를 지원하며, 기본값은 **NHD**.

---

# Ragged Tensor
  **RaggedTensor**는 배치 추론/서빙에서 샘플 간 입력 시퀀스 길이가 다를 때 사용되는 효율적인 데이터 구조.
  ![image](https://github.com/user-attachments/assets/a966f570-80cb-4abf-a396-848f3ece082e)

---

## 사용 사례
  1. Prefill 단계:
     - 시퀀스 길이를 변경할 필요가 없을 때 사용.
     - Key/Value 텐서를 패딩 없이 저장 가능.

  2. 구조:
     - 모든 요청의 Key/Value를 단일 데이터 텐서에 패킹.
     - `indptr` 배열을 사용하여 각 요청의 시퀀스 길이 정보 저장.
       - `indptr`의 크기는 `(num_requests+1)`이며, 첫 번째 요소는 항상 `0`.
     - 데이터 텐서의 Shape:
       - NHD 레이아웃: `(indptr[-1], num_heads, head_dim)`

---

## Key/Value 슬라이싱(Mask Layout 2D Ragged Tensor)

## 데이터 구조

### 1. Query와 KV 길이 저장
  - 요청마다 Query와 KV 길이가 다를 수 있음.
  - 패딩 없이 이를 처리하기 위해 Indptr Arrays를 사용:
    - `qo_indptr` (Query 길이 정보): `(num_requests + 1)` 크기.
      - 요청 `i`의 Query 길이: `qo_indptr[i+1] - qo_indptr[i]`.
    - `kv_indptr` (KV 길이 정보): `(num_requests + 1)` 크기.
      - 요청 `i`의 KV 길이: `kv_indptr[i+1] - kv_indptr[i]`.

---

### 2. Mask Data
  - 모든 요청의 Attention Mask를 Query를 첫 번째 차원, KV를 마지막 차원으로 flatten하여 1D 배열로 저장:
    - Flattened Mask Array (`mask_data`):
      - Shape: `(qk_indptr[-1],)`
      - `qk_indptr`: 요청별 마스크의 시작 위치를 저장하는 배열.
        - `qk_indptr[1:] = cumsum(qo_len * kv_len)`로 계산.

  ![image](https://github.com/user-attachments/assets/3a96e675-413f-4174-98a7-d9ba4f125700)

  - 요청 `i`의 Attention Mask를 슬라이싱:
    ```python
    mask_data[qk_indptr[i]:qk_indptr[i+1]]
    ```

### 3. Bit-packed Mask
  - Boolean Mask 데이터를 압축하여 메모리를 절약:
    - Boolean Mask의 각 요소를 1비트로 표현.
    - 8개의 요소를 uint8 데이터 타입으로 패킹.
    - Little-Endian 비트 순서를 사용.
  - FlashInfer는 Boolean Mask와 Bit-Packed Mask를 모두 지원:
    - Boolean Mask가 제공되면 내부적으로 Bit-Packed Mask로 변환.

---

# Page Table Layout
  Page Table Layout은 KV-Cache가 동적으로 변화하는 상황(예: append 또는 decode 단계)에서 사용됨.  
  이는 시퀀스 길이가 요청마다 시간이 지남에 따라 변하기 때문에 모든 Key/Value를 하나의 텐서에 패킹하는 것이 비효율적이기 때문임.  
  **vLLM**에서는 KV-Cache를 Page Table로 구성하는 방식을 제안했으며,  
  **FlashInfer**에서는 이를 **Block Sparse Matrix**로 간주하고 CSR(Compressed Sparse Row) 형식을 사용하여 페이지를 인덱싱.

  ![image](https://github.com/user-attachments/assets/08eb4479-b977-4184-ba18-7b7ce37aeefc)


## 기본 개념
  - `page_indices`:
    - 각 요청이 사용한 page들의 인덱스를 리스트 형태로 관리.
  - `last_page_len`:
    - 요청별 마지막 페이지에 포함된 토큰 수.
    - `1 <= last_page_len[i] <= page_size`
  - KV 시퀀스 길이 계산:
    - 요청 `i`의 KV 시퀀스 길이 = `page_size * (len(page_indices[i]) - 1) + last_page_length[i]`

---

## 전체 인덱스 배열
  - `kv_indptr` (길이: `num_requests+1`):
    - 모든 요청의 `page_indices`를 연결했을 때의 시작 인덱스를 기록한 배열.
    - 예: `[0, len(page_indices[0]), len(page_indices[0]) + len(page_indices[1]), ...]`

  - `kv_page_indices` (길이: `kv_indptr[-1]`):
    - 모든 요청의 `page_indices`를 단순히 이어붙인 배열.

  - `kv_last_page_lens` (길이: `num_requests`):
    - 모든 요청의 `last_page_len`을 이어붙인 배열.

---

## kv_data 텐서 구조
  - 단일 5D 텐서 형태:
    - NHD Layout:
      ```
      (max_num_pages, 2, page_size, num_heads, head_dim)
      ```
    - HND Layout:
      ```
      (max_num_pages, 2, num_heads, page_size, head_dim)
      ```
    - 여기서 `2`는 K/V 구분.

  - 튜플 형태 (k_data, v_data)로 분리한 4D 텐서:
    - NHD Layout:
      ```
      (max_num_pages, page_size, num_heads, head_dim)
      ```
    - HND Layout:
      ```
      (max_num_pages, num_heads, page_size, head_dim)
      ```

---

# 멀티 레벨(Multi-level) Cascade Inference 데이터 레이아웃
  멀티 레벨 Cascade Inference를 사용할 경우, Query와 Output은 Ragged Tensor 형태로 저장되며,
  모든 레벨의 KV-Cache는 하나의 통합된 Paged KV-Cache 구조로 관리됨.

  각 레벨마다 `qo_indptr`, `kv_page_indptr`, `kv_page_indices`, `kv_last_page_len`이 존재하며,
  이는 Prefix 재사용을 위한 서브트리에 추가되는 토큰 수의 누적합 형태로 관리.

---

## 주요 포인트
  1. Ragged Tensor를 통한 Query/Output 관리
  2. Paged KV-Cache의 통합 관리
  3. 여러 레벨에서 동일한 데이터 레이아웃 공유

---

## 예시 설명
  - 8개의 요청(request)
  - 3개의 레벨로 구성된 Prefix 구조
  - 레벨별로 다른 `qo_indptr` / `kv_page_indptr` 배열을 사용해 동일한 데이터에서 다른 관점 확보
  ![image](https://github.com/user-attachments/assets/ee8ddd91-51cc-49e9-b7d5-9d694c57d185)

---

# FlashInfer 0.2 - 12/16 발표

---

## 주요 개선 사항
  1. Sparse(Page) Attention 가속화: FlashAttention-3 템플릿 도입
     - Hopper GPU용 최적화 (fp8 지원)
  2. JIT(Just-In-Time) 컴파일 지원
     - 다양한 Attention 변형에 대해 JIT 컴파일
  3. Multi-head Latent Attention(MLA) 디코딩 지원
  4. FlashAttention-3 및 Block/Vector-Sparsity 지원
     - page_size=1로 block-sparse attention kernel 구현 가능
  5. CUDAGraph Compatibility for Variable-Length Inputs
     - prefill attention에서 CUDAGraph 사용 가능
  6. torch.compile 호환
     - pytorch custom operators standard 준수
  7. Non-Contiguous KV-cache support
     - offloading 가능

FlashInfer roadmap : https://github.com/flashinfer-ai/flashinfer/issues/675
## FlashAttention-3 및 Block/Vector-Sparsity 지원
- page_size=1 일 때의 PageAttention operator 를 지원함 -> 이를 이용해 block-sparse attention kernel을 구현


## CUDAGraph Compatibility for Variable-Length Inputs
- prefill attention 에서도 CUDAGraph 를 사용할 수 있도록 함.


## torch.compile 호환 
- pytorch custom operators standard 를 지켜서 만듬

## Non-Contiguous KV-cache support
- KV-Cache 가 non-contiguous storage layout 을 사용할 수 있음 - offloading 가능


FlashInfer roadmap : https://github.com/flashinfer-ai/flashinfer/issues/675
