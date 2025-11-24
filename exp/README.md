# Mobility Experiments

이 폴더는 `uv` 기반으로 실행할 수 있도록 구성되어 있습니다.

## 사전 준비

1. [uv](https://github.com/astral-sh/uv)를 설치합니다. (예: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. Python 3.10 이상이 PATH에 있어야 합니다.

## 의존성 설치

```bash
cd draft/exp
uv sync
```

`uv`가 가상환경을 만들고 `pyproject.toml`에 정의된 패키지(`torch`, `numpy`, `pandas`, `scikit-learn` 등)를 설치합니다.

## 실험 실행

기본으로는 `../data`(예: `/Users/.../draft/data`)에서 `IFs_*.csv`, `timedata_*.csv`를 찾습니다.

```bash
uv run mobility-exp --user kdy
```

다른 위치에 데이터가 있다면 경로를 직접 지정하세요.

```bash
uv run mobility-exp --data-dir /path/to/data --user kdy
uv run mobility-exp --imf-path /path/to/IFs_xxx.csv --raw-path /path/to/timedata_xxx.csv
```

첫 번째 줄처럼 `--data-dir`로 폴더를 지정하면 내부 파일명이 자동으로 매칭되고, 두 번째 줄처럼 파일 경로를 각각 직접 줄 수도 있습니다.

실행이 끝나면 각 모델의 성능 지표가 터미널에 출력되고 `experiment_metrics.csv`에도 저장됩니다.
