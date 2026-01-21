# 최적화 파츠 구분 문서

이 문서는 VPL-GP 시스템의 주요 컴포넌트를 파츠별로 구분하여 최적화 포인트를 찾기 쉽게 구성했습니다.

## 📊 파츠별 코드 라인 수

### 1. Trainer 파츠 (가장 큰 파츠)
- **vpl_reward_choice_trainer.py** (902 lines) - VPL binary selector trainer
- **reward_trainer.py** (526 lines) - DPO reward trainer
- **vpl_gp_reward_choice_trainer.py** (169 lines) - VPL-GP trainer
- **reward_choice_trainer.py** (196 lines) - Baseline binary selector trainer
- **trainer.py** (382 lines) - Base LLM trainer

### 2. RLHF 파츠
- **standalone_training.py** (651 lines) - Standalone RLHF training (data generation + DPO)
- **variational_selector.py** (379 lines) - Variational selection for preference data
- **load_vpl_components.py** (187 lines) - VPL components loading utility
- **main.py** (97 lines) - RLHF entry point

### 3. Model 파츠
- **adapter_builder.py** (394 lines) - Adapter model wrapper (LoRA)
- **variational_encoder_gp.py** (238 lines) - Variational encoder with GP prior
- **variational_encoder.py** (197 lines) - Base variational encoder
- **model_builder.py** (122 lines) - Model loading and initialization

### 4. Server/Client 파츠
- **server.py** (llm_local/server.py) - Server-side logic (z collection, aggregation, t-SNE)
- **client.py** (llm_local/client.py) - Client-side logic (training, z transmission)

### 5. Data Loading 파츠
- **llm_dataset.py** - Dataset classes (LLMDataset, LLMComparisonDataset)
- **dataloader/** - Data loading utilities (hh_rlhf, reddit_tldr, etc.)

### 6. Metrics 파츠
- **hhrl_metrics.py** - Harmlessness/Helpfulness reward model evaluation
- **winrate_metrics.py** - Winrate calculation
- **vpl_metrics.py** - VPL-specific metrics (KL loss, reconstruction loss, orthogonal loss)

### 7. Visualization 파츠
- **z_visualization.py** - t-SNE visualization of latent z values

---

## 🔍 파츠별 최적화 포인트

### 파츠 1: Trainer
**파일:**
- **vpl_reward_choice_trainer.py** (902 lines) - VPL binary selector trainer
- **vpl_gp_reward_choice_trainer.py** (169 lines) - VPL-GP trainer (extends VPL)
- **reward_trainer.py** (526 lines) - DPO reward trainer
- **reward_choice_trainer.py** (196 lines) - Baseline binary selector trainer

**주요 기능:**
- Forward pass with variational inference
- KL divergence calculation (with GP prior for VPL-GP)
- Orthogonal loss computation
- Feature extraction from LLM embeddings
- Z posterior inference
- DPO loss calculation (for RLHF)

**최적화 가능 영역:**
1. **Feature Extraction**: 중복 forward pass 제거, 배치 처리
2. **KL Loss 계산**: 배치 처리 최적화, 효율적인 mixture prior 계산
3. **Orthogonal Loss**: 행렬 연산 최적화, 프로토타입 업데이트 최적화
4. **Memory Management**: 중간 텐서 메모리 해제, gradient checkpointing
5. **Forward Pass**: 불필요한 계산 제거, 연산 그래프 최적화

### 파츠 2: RLHF Training
**파일:**
- **standalone_training.py** (651 lines) - Standalone RLHF training
- **variational_selector.py** (379 lines) - Variational selection for preference data
- **load_vpl_components.py** (187 lines) - VPL components loading utility

**주요 기능:**
- Pairwise data generation (with z-dependent generation)
- Variational selection for preference data
- DPO training data preparation
- VPL components loading (encoder, feature extractor, z_to_embedding)

**최적화 가능 영역:**
1. **Data Generation**: 배치 생성 최적화, 병렬 생성
2. **Z Inference**: 중복 계산 제거, 배치 추론
3. **Memory Usage**: 생성된 텍스트 메모리 관리, 스트리밍 처리
4. **Component Loading**: 체크포인트 로딩 최적화, 지연 로딩
5. **Selection**: 효율적인 variational selection 알고리즘

### 파츠 3: Model Components
**파일:**
- **adapter_builder.py** (394 lines) - Adapter model wrapper (LoRA)
- **variational_encoder_gp.py** (238 lines) - Variational encoder with GP prior
- **variational_encoder.py** (197 lines) - Base variational encoder
- **model_builder.py** (122 lines) - Model loading and initialization

**주요 기능:**
- Posterior encoding (mu, logvar)
- Prior update (mixture of other clients for GP)
- KL divergence with mixture prior
- Adapter management (LoRA)
- Model loading and initialization

**최적화 가능 영역:**
1. **Prior Update**: 효율적인 mixture 계산, 배치 업데이트
2. **Reparameterization**: 메모리 효율적인 샘플링, in-place 연산
3. **Gumbel Softmax**: Temperature annealing 최적화, 수치 안정성
4. **Adapter Management**: 효율적인 adapter 활성화/비활성화
5. **Model Loading**: 지연 로딩, 메모리 효율적 로딩

### 파츠 4: Server/Client Logic
**파일:**
- **server.py** (922 lines) - Server-side logic
- **client.py** (279 lines) - Client-side logic
- **z_visualization.py** (182 lines) - t-SNE visualization
- **aggregator.py** (168 lines) - Model aggregation

**주요 기능:**
- Z distribution collection and broadcasting
- Orthogonal label computation (k-means)
- t-SNE visualization
- Model aggregation (FedAvg)
- Client training coordination

**최적화 가능 영역:**
1. **Z Collection**: 통신 오버헤드 감소, 배치 처리
2. **Visualization**: 주기적 업데이트 최적화, 메모리 관리
3. **Label Computation**: K-means 최적화, 초기화 전략
4. **Aggregation**: 효율적인 파라미터 병합, 메모리 효율적 집계
5. **Client Coordination**: 비동기 처리, 효율적인 메시지 전달

### 파츠 5: Data Loading
**파일:**
- **llm_dataset.py** (220 lines) - Dataset classes
- **dataloader/** - Data loading utilities (hh_rlhf, reddit_tldr, etc.)

**주요 기능:**
- Dataset preprocessing
- Tokenization
- Batch preparation
- Z value storage (for RLHF)

**최적화 가능 영역:**
1. **Tokenization**: 배치 토크나이제이션, 캐싱
2. **Data Caching**: 중복 로딩 방지, 메모리 효율적 캐싱
3. **Memory Mapping**: 대용량 데이터 처리
4. **Z Storage**: 효율적인 z 값 저장/로딩

---

## 📋 최적화 우선순위

### High Priority (성능 영향 큼)
1. **Trainer Forward Pass** - 가장 자주 호출됨
2. **Data Generation** - RLHF에서 시간 소모 큼
3. **Feature Extraction** - 중복 계산 가능성

### Medium Priority (메모리 최적화)
1. **Variational Encoder** - 메모리 사용량 큼
2. **Server Z Collection** - 통신 오버헤드
3. **Model Loading** - 초기화 시간

### Low Priority (코드 정리)
1. **Metrics Calculation** - 평가 시에만 실행
2. **Visualization** - 주기적 실행
3. **Utility Functions** - 드물게 호출

---

## 🎯 다음 단계

각 파츠별로 상세 분석을 진행하여 구체적인 최적화 포인트를 찾아야 합니다.
