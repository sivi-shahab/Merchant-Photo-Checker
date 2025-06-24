# Merchant-Photo-Checker
# 📘 Project Setup with Docker & Ollama

This README guides you through setting up and running the project using Docker Compose, configuring Ollama for vision model inference, and preparing the environment.

---
## Step 0: Install Docker and Docker Compose

## 🚀 Step 1: Build & Start the Project

change mongodb and postgres setting in `.env` file in the project root directory and fill it with your production environment settings. Here's an example:

```
# .env prod
MONGO_URI=mongodb://user1:c0B5%40!2023%23@172.18.46.191:27017/merchant?authSource=merchant

POSTGRES_HOST=10.15.42.168
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_USER=datamgmt
POSTGRES_PASSWORD=D@t44mgmt!123
```
> ✅ Ensure Docker and Docker Compose are installed on your machine.

---

## 🐳 Step 2: Access the Ollama Container

```bash
docker-compose up --build
```

This command will:

- Build all services defined in `docker-compose.yml`
- Launch the containers


## 📥 Step 3: Pull the LLaMA Vision Model

After the containers are running, open a terminal and execute:

```bash
docker exec -it ollama bash
```

This gives you an interactive shell inside the `ollama` container.

---


## ⚙️ Step 4: Configure Environment Variables
Once inside the Ollama container, run the following command:

```bash
ollama pull llama3.2-vision
```

This will download the `llama3.2-vision` model required for image-based inference.

---



> 🔐 **Note:** Never commit your `.env` file to version control.

---

## ✅ You're Ready!

- The application is now running in Docker
- Ollama is set up with the vision model
- Your environment is configured for production

Feel free to test the API or connect your application to the running backend.

http://{ip_server}:8000/docs

---

## 📎 Additional Tips

- To stop all containers: `docker-compose down`
- To view logs: `docker-compose logs -f`
- To rebuild only one service: `docker-compose build <service_name>`

