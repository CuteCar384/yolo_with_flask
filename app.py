import shutil
from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
import glob

from ultralytics import YOLO

def clear_directory(directory_path):
    # 遍历目录下的文件和子目录
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # 判断是文件还是子目录
        if os.path.isfile(file_path):
            # 如果是文件，就删除
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # 如果是子目录，递归删除子目录
            clear_directory(file_path)
            os.rmdir(file_path)

    print(f"All files in {directory_path} have been deleted.")
def extract_trailing_number(directory_name, prefix):
    # 从目录名称中提取尾号的数字部分
    number_part = directory_name[len(prefix):]

    try:
        # 尝试将提取的数字部分转换为整数
        return int(number_part)
    except ValueError:
        # 如果转换失败，则返回一个特殊值，表示无效
        return float('-inf')  # 一个负无穷小值


def get_latest_image_in_directories(base_directory, prefix='predict'):
    # 找到所有以 'predict' 为前缀的目录
    predict_directories = [d for d in os.listdir(base_directory) if d.startswith(prefix)]
    #print('predictfolders:', predict_directories)
    if not predict_directories:
        print("No directories found.")
        return None

    # 找到尾号最大的目录
    latest_directory = max(predict_directories, key=lambda d: extract_trailing_number(d, prefix))
    #print('latest_directory:', latest_directory)

    # 构建最新目录的完整路径
    latest_directory_path = os.path.join(base_directory, latest_directory)

    # 获取最新目录下所有文件
    latest_images = get_latest_image(latest_directory_path)

    return latest_images

def save_image_to_directory(image_path, destination_directory):
    try:
        # 获取图片文件名
        image_filename = os.path.basename(image_path)

        # 构建目标路径
        destination_path = os.path.join(destination_directory, image_filename)

        # 使用shutil复制文件
        shutil.copyfile(image_path, destination_path)

        print(f"Image successfully saved to: {destination_path}")
    except Exception as e:
        print(f"Error saving image: {e}")


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the 'results' folder exists
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

def get_latest_image(directory):
    # 获取目录下所有文件
    files = os.listdir(directory)

    # 过滤出图片文件（你可能需要根据实际情况进行调整）
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 获取文件的完整路径
    image_paths = [os.path.join(directory, file) for file in image_files]

    # 获取文件的创建时间
    creation_times = [os.path.getctime(path) for path in image_paths]

    # 找到最新的一张图片
    latest_image_path = image_paths[creation_times.index(max(creation_times))]

    return latest_image_path


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 使用示例
    directory_to_clear = 'runs/segment'
    clear_directory(directory_to_clear)

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # 读取文件内容
        file_content = file.read()

        # 保存到uploads目录
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        with open(filename, 'wb') as upload_file:
            upload_file.write(file_content)

        # 获取下拉框的值
        # 改成yolo多个模型
        model_name = request.form.get('model_option')
        # Load a model
        model = YOLO(model_name)  # pretrained YOLOv8n model

        # 调用函数的地方
        results = model.predict(f'{filename}',save=True,device=0)  # return a list of Results objects
        # 传递基础目录和前缀
        base_directory = 'runs/segment'
        latest_images = get_latest_image_in_directories(base_directory)
        save_image_to_directory(f'{latest_images}','./static/results')

        return render_template('index.html', filename=os.path.basename(filename), processed_image=os.path.basename(latest_images))
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)
