import logging
import os

import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

from core import check_onnx

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


# 아래와 같이 사이즈가 다른 크기를 onnx로 export하는 경우 anchor는 못쓰게 됨.
def onnx_export(model="512_512_SGD_PVGG16_512", load_period=100, input_size=(512, 1024), dtype=np.float32,
                onnx_path=""):
    input_shape = (1, 3) + input_size
    weight_path = f"weights/{model}"

    sym = os.path.join(weight_path, f'{model}-symbol.json')
    params = os.path.join(weight_path, f'{model}-{load_period:04d}.params')

    temp = model.split("_")
    onnx_name = str(input_size[0]) + "_" + str(input_size[1]) + "_" + temp[2] + "_" + temp[3]
    onnx_file_path = os.path.join(onnx_path, f"{onnx_name}.onnx")

    try:
        onnx_mxnet.export_model(sym=sym, params=params, input_shape=[input_shape], input_type=dtype,
                                onnx_file_path=onnx_file_path, verbose=False)
    except Exception as E:
        logging.error(f"ONNX model export 예외 발생 : {E}")
    else:
        logging.error(f"ONNX model export 성공")

    try:
        check_onnx(onnx_path)
        logging.info(f"{os.path.basename(onnx_path)} saved completed")
    except Exception as E:
        logging.error(f"ONNX model check 예외 발생 : {E}")
    else:
        logging.info("ONNX model check completed")


if __name__ == "__main__":
    onnx_export(model="512_512_SGD_PVGG16_512",
                load_period=100,
                input_size=(768, 1280),
                dtype=np.float32,
                onnx_path="")
