# CRYPTO TREND PREDICTOR

`PYTORCH TRANSFORMER` | `TIME-SERIES FORECASTING` | `FASTAPI` | `STREAMLIT`

An end-to-end deep learning forecasting system built on a custom PyTorch Transformer architecture to analyze and predict cryptocurrency price trends. Designed for rapid cloud deployment and real-time market inference.

---

## OVERVIEW & PURPOSE

Cryptocurrency markets operate continuously with extreme volatility, abrupt momentum shifts, and complex non-linear patterns. Traditional statistical models and simple moving averages often fail during sudden reversals because they evaluate historical prices in isolation rather than capturing long-range temporal dependencies.

This project addresses that limitation by applying **Transformer self-attention mechanisms** to historical market sequences. By dynamically weighing the significance of past price intervals across multiple time steps, the model identifies underlying trend direction and structural market shifts with higher contextual awareness.

---

## KEY BENEFITS

* **Attention-Driven Sequence Modeling:** Captures long-term dependencies and short-term volatility spikes simultaneously using multi-head self-attention in PyTorch.
* **Ready-to-Deploy Inference:** Ships with pre-trained weights (`crypto_transformer.pth`) for immediate forecasting without requiring computationally heavy retraining.
* **Cloud-Native Dashboard:** Built for seamless live hosting on Streamlit Community Cloud, allowing real-time interactive trend visualization directly in the browser.
* **Dual-Architecture Flexibility:** Supports both a standalone interactive Streamlit application (`app.py`) and a decoupled FastAPI + HTML stack (`main.py` and `index.html`) for custom API integrations.
* **Modular Engineering:** Cleanly separated training, evaluation, model definition, and serving modules for straightforward customization across new tokens or timeframes.

---

## SYSTEM ARCHITECTURE

```text
+-------------------+      +------------------------+      +-----------------------+
|   Market Data     | ---> |  PyTorch Transformer   | ---> |   Trend Forecast &    |
|    (yfinance)     |      | (crypto_transformer)   |      |  Confidence Metrics   |
+-------------------+      +------------------------+      +-----------------------+
                                                                       |
                                                    +------------------+------------------+
                                                    |                                     |
                                                    v                                     v
                                         +---------------------+               +---------------------+
                                         |  Streamlit Cloud UI |               | FastAPI + Web Front |
                                         |      (app.py)       |               | (main.py/index.html)|
                                         +---------------------+               +---------------------+


AUTHOR
Yousaf Shiraz

AI Engineer & Full-Stack Developer

GitHub: github.com/Myousaf-arch

