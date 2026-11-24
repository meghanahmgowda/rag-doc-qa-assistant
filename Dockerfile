RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user 

PATH=/home/user/.local/bin:$PATH 

HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && 

pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN mkdir -p uploaded_docs vector_store

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]