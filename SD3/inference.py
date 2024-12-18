import os
from diffusers import StableDiffusion3Pipeline
import torch

# 初始化 StableDiffusionPipeline
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe = pipe.to("cuda")

# 定义生成图片函数
def ImageGen(prompt, index):
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
        height = 480,
        width = 640,
    ).images[0]
    return image

# 读取文本文件并生成图片，并记录图片路径
def generate_images_from_text(txt_file):
    folder = "ImageGen"
    
    # 创建文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 创建或覆盖一个用于保存图片路径的 txt 文件
    with open("image_paths.txt", "w", encoding="utf-8") as path_file:
        
        # 打开txt文件，逐行读取
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            for idx, line in enumerate(lines):
                prompt = line.strip()
                if prompt:  # 确保不是空行
                    image = ImageGen(prompt, idx)
                    image_path = os.path.join(folder, f"image_{idx+1}.png")
                    image.save(image_path)
                    
                    # 把图片路径写入到txt文件中
                    path_file.write(f"{image_path}\n")
                    print(f"Saved image_{idx+1}.png for prompt: {prompt}")

# 使用示例
txt_file = "/afs/crc.nd.edu/user/z/zyuan2/code/SD3/enhancement.txt"  # 替换为你的txt文件路径
generate_images_from_text(txt_file)
