# Application setup

## Backend

1. Create a `venv` environment using following command `python -m venv <venv_name>`. Make sure to create this virtual 
environment on the same level as the `LLM_writers` repository in order for the frontend to work.
2. Install all the required `python` packages. They are listed in `requirements.txt` file, that is present in
`LLM_writers` directory. It can be done using `pip install -r requirements.txt`.
3. The application can be started using the following command `uvicorn LLM_writers.controller:app --reload`. However, 
some environment variables need to be setup for the app to properly access the database and retrieve the embeddings.
Those variables are: `QDRANT_API_KEY` and `QDRANT_CLOUD_URL`.

[//]: # (4. For the best results and easiest running, a startup configuration can be created)

## Frontend

1. Head to `llm_fe` directory and the install all the dependencies using `npm install`. 
2. Run the application using `npm run dev`. Then the application should be run at `http://localhost:5173/`.