mlflow server \
    --backend-store-uri postgresql+psycopg2://manipurl_user:123@localhost:5432/manipurl \
    --default-artifact-root file:/home/$USER/mlruns \
    --host 0.0.0.0 \
    --port 5000