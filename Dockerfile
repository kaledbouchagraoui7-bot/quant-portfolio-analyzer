FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir pytest
CMD ["python", "quant_portfolio_analyzer.py", "--input", "sample_prices.csv", "--periods-per-year", "12", "--format", "md"]
