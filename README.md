# geospectra

`geospectra` 提供了一组用于地学建模的基函数和简单工具，方便在机器学习流程中使用球冠谐模型、二次多项式等设计矩阵。

## 特性

- **PolynomialBasis**：构造二元多项式设计矩阵，可选择普通、切比雪夫或勒让德基。
- **SphericalHarmonicsBasis**：生成球面（或球冠）谐函数特征，支持经纬度自动转换。
- **LinearRegressionCond**：在线性回归中增加条件数阈值控制，提升数值稳定性。
- **PCA**：简化的主成分分析实现，可按需获取设计矩阵。
- 内置 `main()` 入口函数，可在命令行调用 `python -m geospectra`。

## 安装

项目依赖 Python 3.10 以上环境，推荐使用 [uv](https://github.com/astral-sh/uv) 进行安装：

```bash
uv pip install -e .
```

或者使用 `pip`：

```bash
python -m pip install -e .
```

开发环境可通过 `.[dev]` 额外安装 `pytest` 与 `ruff` 等工具。

## 快速示例

以下示例展示如何在 `scikit-learn` 管道中使用多项式基函数进行回归：

```python
import numpy as np
from sklearn.pipeline import Pipeline
from geospectra import PolynomialBasis, LinearRegressionCond

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(100, 2))
y = X[:, 0] ** 2 + X[:, 1]  # 构造示例目标

model = Pipeline([
    ("basis", PolynomialBasis(degree=2)),
    ("reg", LinearRegressionCond()),
])
model.fit(X, y)
pred = model.predict(X)
```

若需要使用球面谐函数特征，只需替换管道中的基函数生成器：

```python
from geospectra import SphericalHarmonicsBasis

model = Pipeline([
    ("basis", SphericalHarmonicsBasis(degree=3, include_bias=False)),
    ("reg", LinearRegressionCond()),
])
```

## 运行测试

仓库使用 `ruff` 进行代码格式化和静态检查，测试框架为 `pytest`：

```bash
ruff format .
ruff check .
pytest
```

以上命令仅在代码修改后需要执行，更新文档时可跳过。

