FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the artifacts at image build time if the raw data is present
RUN python -m src.train || echo "No raw data at build time - run src.train later"

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
