import os 
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from llama_index import GPTSimpleVectorIndex
import uvicorn
import logging

# Load environment variables from .env file
# load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

src_path = os.getcwd() + "/vector_index_gpt-3.5-turbo_v2.json"

ai_bot = GPTSimpleVectorIndex.load_from_disk(src_path)


@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/getChatBotResponse")
def get_bot_response(msg: str):
    data = dict()
    response = ai_bot.query(msg, response_mode="compact")
    data["response"] = str(response.response)
    data["sources"] = [f"https://hocvien.tiki.vn/faq/{source_node.extra_info.get('source')}/" for source_node in response.source_nodes]
    return data

if __name__ == "__main__":
    uvicorn.run("main:app")