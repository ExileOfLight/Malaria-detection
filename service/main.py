import starlette
from fastapi import FastAPI
from service.api.api import main_router
import onnxruntime as rt
import uvicorn
from starlette.responses import HTMLResponse

app = FastAPI(project_name="Malaria Detection")
app.include_router(main_router)

model_path = "service/eff_quantized.onnx"
providers = ['CPUExecutionProvider']
m_q = rt.InferenceSession(
    model_path, providers=providers
)


@app.get("/")
async def root():
    message = "Simple web app that runs a trained and quantized neural network (Transfer learning with Pretrained " \
              "Efficient Net).<br/>Use /docs for documentation and testing.<br/>Full code for this neural network and "\
              "additional studied materials:<br/>https://colab.research.google.com/drive" \
              "/1_ro6_k7KzdhXtxciTuuwa9_pfn8dTg2o?usp=sharing<br/>"
    html_content = html_content = f"""
    <html>
    <head>
        <style>
            body {{ background-color: #222222; color: white; }}
        </style>
    </head>
    <body>
        <p>{message}</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
