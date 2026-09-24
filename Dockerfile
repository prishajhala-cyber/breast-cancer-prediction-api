# Match this to your local Python version (check with: python --version)
FROM python:3.12-slim

# Hugging Face Spaces runs containers as a non-root user with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Install dependencies first. Docker caches this layer, so rebuilding after a
# code change skips the slow install step unless requirements.txt changed.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the project (anything in .dockerignore is skipped)
COPY --chown=user . .

# Train the model inside the image, so it always matches the installed
# scikit-learn version. The dataset ships with scikit-learn, so no download.
RUN python model/train.py

# Quality gate: the build fails if any test fails
RUN python -m pytest -q -p no:cacheprovider

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]