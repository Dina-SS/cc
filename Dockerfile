FROM python:3.8-slim
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./


# Install dependencies:
COPY requirements.txt .
RUN python3 -m pip install --default-timeout=100 -r requirements.txt
RUN python3 -m pip install Flask gunicorn

RUN python3 -m spacy download en
RUN python3 -m spacy download en_core_web_sm

RUN [ "python3", "-c", "import nltk; nltk.download('punkt')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('wordnet')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('averaged_perceptron_tagger')" ]


CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flask_app:app

