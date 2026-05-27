# Gemma 4 (SageMaker JumpStart) + OpenAI 호환 API — End-to-End 가이드

Gemma 4 모델을 SageMaker JumpStart에 배포하고, OpenAI 호환 API로 호출하는 전체 과정을 처음부터 끝까지 정리한 문서입니다.

참고 블로그
- Gemma 4 on SageMaker JumpStart: https://aws.amazon.com/about-aws/whats-new/2026/04/gemma-4-models-on-sagemaker-jumpstart/
- OpenAI 호환 API 지원: https://aws.amazon.com/blogs/machine-learning/announcing-openai-compatible-api-support-for-amazon-sagemaker-ai-endpoints/

---

## 0. 사전 준비

- AWS 자격증명 설정 (SageMaker 배포 + `sagemaker:InvokeEndpoint` + `sagemaker:CallWithBearerToken` 권한)
- SageMaker 실행 역할(Execution Role) ARN
- 배포할 인스턴스의 엔드포인트 사용 쿼터 (E4B는 `ml.g6e.2xlarge` 1개 필요)

Gemma 4 변형과 기본 인스턴스:

| 모델 ID | 기본 인스턴스 | 비고 |
|---|---|---|
| `huggingface-vlm-gemma-4-e4b-it` | `ml.g6e.2xlarge` | 가장 저렴 (이 가이드 기준) |
| `huggingface-vlm-gemma-4-26b-a4b-it` | `ml.g6.12xlarge` | 중간 MoE |
| `huggingface-vlm-gemma-4-31b-it` | `ml.g6.24xlarge` | 가장 큼 |

---

## 1. 환경 구성

```bash
cd /Users/hasunyu/Code/gemma
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # sagemaker>=3.12, openai>=2.0, httpx, boto3
```

---

## 2. 설정 확인 — `config.py`

모델 ID, 인스턴스 타입, 리전, 실행 역할 ARN, 엔드포인트 이름을 한곳에서 관리합니다.
환경변수로도 덮어쓸 수 있습니다. 다른 변형/역할을 쓰려면 이 파일만 수정하면 됩니다.

```bash
# 기본값을 그대로 쓰거나, 필요시 환경변수로 변경
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account>:role/service-role/<your-sagemaker-role>"
export GEMMA4_MODEL_ID="huggingface-vlm-gemma-4-e4b-it"     # 선택: 다른 변형
export GEMMA4_INSTANCE_TYPE="ml.g6e.2xlarge"
```

---

## 3. 배포 — `deploy_gemma4.py`

SageMaker Python SDK v3의 `ModelBuilder` 워크플로(`build()` → `deploy()`)를 사용합니다.
Gemma는 게이트 모델이므로 `accept_eula=True`가 필수입니다.

```bash
python deploy_gemma4.py
```

핵심 코드 (스크립트 내부):

```python
from sagemaker.serve.model_builder import ModelBuilder

builder = ModelBuilder(
    model="huggingface-vlm-gemma-4-e4b-it",   # JumpStart 모델 ID
    role_arn=ROLE_ARN,
    instance_type="ml.g6e.2xlarge",
)
builder.accept_eula = True          # 게이트 모델 라이선스 동의 (필수)
builder.build()                     # JumpStart 아티팩트/컨테이너 해석 → Model 생성
builder.deploy(
    endpoint_name="gemma4-e4b-openai-demo",
    initial_instance_count=1,
    instance_type="ml.g6e.2xlarge",
    accept_eula=True,
    wait=False,                     # 일시적 네트워크 오류에 강하도록 자체 폴링으로 대기
)
```

- 배포는 보통 **10~15분** 소요됩니다.
- 완료되면 엔드포인트 이름이 `endpoint_info.json`에 기록되어 테스트/정리 스크립트가 참조합니다.

---

## 4. OpenAI 호환 API로 테스트 — `test_openai_api.py`

```bash
python test_openai_api.py
```

핵심 포인트:

- **Base URL**: `https://runtime.sagemaker.<REGION>.amazonaws.com/endpoints/<ENDPOINT>/openai/v1`
- **인증**: `sagemaker.core.token_generator.generate_token`로 만든 단기 베어러 토큰. `httpx.Auth`로 요청마다 갱신하면 장시간 세션에서도 만료되지 않음.
- **`model` 필드**: JumpStart vLLM 컨테이너는 모델을 `/opt/ml/model`로 서빙하므로, **빈 문자열(`""`)** 을 넣으면 단일 서빙 모델로 자동 라우팅됨. (JumpStart 모델 ID를 그대로 넣으면 404 발생)

핵심 코드:

```python
from datetime import timedelta
import httpx
from openai import OpenAI
from sagemaker.core.token_generator import generate_token

REGION = "us-east-1"
ENDPOINT = "gemma4-e4b-openai-demo"

class SageMakerBearerAuth(httpx.Auth):
    def auth_flow(self, request):
        token = generate_token(region=REGION, expiry=timedelta(minutes=5))
        request.headers["Authorization"] = f"Bearer {token}"
        yield request

client = OpenAI(
    base_url=f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT}/openai/v1",
    api_key=generate_token(region=REGION, expiry=timedelta(minutes=5)),
    http_client=httpx.Client(auth=SageMakerBearerAuth(), timeout=120.0),
)

# 비스트리밍
resp = client.chat.completions.create(
    model="",   # 빈 문자열 = 엔드포인트의 단일 서빙 모델
    messages=[{"role": "user", "content": "What is Amazon SageMaker JumpStart?"}],
    max_tokens=128, temperature=0.2,
)
print(resp.choices[0].message.content)

# 스트리밍
stream = client.chat.completions.create(
    model="", messages=[{"role": "user", "content": "Count 1 to 5."}],
    max_tokens=64, stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 5. 정리 (과금 중단) — `cleanup.py`

엔드포인트는 `InService` 상태인 동안 시간당 과금됩니다 (E4B / `ml.g6e.2xlarge` ≈ $2.2/시간, us-east-1 온디맨드).
테스트가 끝나면 반드시 정리하세요.

```bash
python cleanup.py   # 엔드포인트 + 엔드포인트 구성 + 모델 삭제
```

---

## 전체 흐름 요약

```bash
cd /Users/hasunyu/Code/gemma
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python deploy_gemma4.py     # 1) 배포 (~10-15분)
python test_openai_api.py   # 2) OpenAI 호환 API 테스트 (채팅 + 스트리밍)
python cleanup.py           # 3) 정리 (과금 중단)
```
