import uvicorn
from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
import cv2
import numpy as np

app = FastAPI()

model_new = keras.models.load_model('action.h5')


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

@app.post("/process_frame")
async def process_frame(frame: UploadFile = File(resp)):
    # 读取视频帧数据
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 在这里进行视频帧的处理逻辑
    # 模型预测 导出
    print(111)

    # 返回处理后的结果（可选）
    return {"message": "Frame processed successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
