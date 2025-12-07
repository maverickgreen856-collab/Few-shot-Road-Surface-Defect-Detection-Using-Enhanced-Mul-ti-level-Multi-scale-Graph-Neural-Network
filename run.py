import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from scipy.fftpack import dct, idct
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import logging
from torchvision import transforms
from scipy import ndimage

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置参数
class Config:
    data_dir = "data"  # 原始数据目录
    result_dir = "result"  # 增强结果目录
    debug_dir = "debug"  # 调试输出目录
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    
    # 扩展的天气条件增强提示词 - 更具体的天气描述
    weather_conditions = [
        "heavy rain pouring down, wet reflective surfaces, water droplets", 
        "thick fog covering the scene, low visibility, atmospheric haze",
        "dark night scene, low light conditions, dimly lit environment",
        "heavy snowfall, snow-covered surfaces, winter conditions",
        "bright sunny day, strong sunlight casting sharp shadows",
        "stormy weather with dramatic clouds, thunder and lightning",
        "cloudy overcast day, soft diffuse lighting, no harsh shadows",
        "misty morning, light mist in the air, dewy surfaces",
        "golden hour sunset, warm orange lighting, long shadows",
        "early dawn, soft morning light, cool blue tones",
        "hazy atmosphere, air pollution, reduced visibility",
        "freezing conditions, ice crystals, frosty surfaces"
    ]
    
    # 场景变换提示词 - 更具体的场景描述
    scene_modifiers = [
        "urban street with buildings",
        "natural environment with vegetation",
        "industrial area with machinery",
        "road surface with markings",
        "pavement with cracks"
    ]
    
    # 视角变换提示词
    perspective_modifiers = [
        "close-up view emphasizing details",
        "wide angle shot showing context",
        "normal perspective"
    ]
    
    # 分形参数空间
    fractal_params = [
        {"xmin": -2.0, "xmax": 1.0, "ymin": -1.5, "ymax": 1.5},
        {"xmin": -1.0, "xmax": -0.5, "ymin": 0.0, "ymax": 0.5},
        {"xmin": -0.75, "xmax": -0.74, "ymin": 0.10, "ymax": 0.11},
        {"xmin": -0.745, "xmax": -0.744, "ymin": 0.105, "ymax": 0.106},
    ]
    
    alpha = 0.12  # 提高分形融合系数
    max_iter = 100  # 分形迭代次数
    pyramid_levels = 3  # 金字塔融合层级
    max_retries = 5  # 增加重试次数
    retry_delay = 5  # 重试延迟(秒)
    mask_train_epochs = 2  # 减少掩码训练轮数
    mask_lr = 0.01  # 掩码训练学习率
    lambda_reg = 0.1  # 掩码正则化系数
    min_image_size = 256  # 最小图像尺寸
    image_size = 224  # 统一处理尺寸
    max_strength = 0.5  # 提高扩散强度，增加变化
    min_strength = 0.3  # 提高扩散强度，增加变化
    fractal_alpha_range = (0.08, 0.15)  # 提高分形融合系数范围
    min_similarity = 0.25  # 语义相似度阈值
    debug_mode = True  # 启用调试模式
    debug_samples_per_class = 5  # 每个类别随机选择5张图片保存调试信息
    max_style_intensity = 0.4  # 风格迁移最大强度
    min_style_intensity = 0.2  # 风格迁移最小强度
    classes = ["S1", "S2", "S3"]  # 只处理S1-S3类别

# 确保目录存在
os.makedirs(Config.data_dir, exist_ok=True)
os.makedirs(Config.result_dir, exist_ok=True)
os.makedirs(Config.debug_dir, exist_ok=True)  # 创建调试目录

for class_name in Config.classes:
    os.makedirs(os.path.join(Config.data_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(Config.result_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(Config.debug_dir, class_name), exist_ok=True)  # 为每个类别创建调试目录

# 简化的掩码生成器模型
class MaskGenerator(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        original_size = x.shape[2:]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        return self.sigmoid(x)

# 初始化模型
def init_models():
    logging.info("加载稳定扩散模型...")
    # 使用在线模型，自动下载
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if "cuda" in Config.device else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(Config.device)
    
    logging.info("加载CLIP模型...")
    # 使用在线模型，自动下载
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(Config.device)
    
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    
    mask_gen = MaskGenerator().to(Config.device)
    
    return pipe, clip_model, clip_processor, mask_gen

# 生成包含天气、场景和视角变化的提示词 - 更强调天气变化
def generate_labeled_prompt(orig_label):
    weather = random.choice(Config.weather_conditions)
    
    # 减少场景和视角变化的频率和强度
    use_scene = random.random() < 0.3  # 30%概率使用场景变化
    use_perspective = random.random() < 0.2  # 20%概率使用视角变化
    
    scene = random.choice(Config.scene_modifiers) if use_scene else ""
    perspective = random.choice(Config.perspective_modifiers) if use_perspective else ""
    
    # 构建提示词，重点强调天气变化和缺陷保留
    components = [weather]
    if scene:
        components.append(scene)
    if perspective:
        components.append(perspective)
    
    # 确保缺陷特征(如坑洞、裂缝、龟裂)在提示词中保持清晰
    components.append(f"{orig_label}, clearly visible cracks and potholes, detailed surface texture, high detail, sharp focus, realistic")
    
    return ", ".join(components)

# 生成更明显的分形图像
def generate_natural_fractal(height, width, max_iter=Config.max_iter):
    # 随机选择分形参数
    params = random.choice(Config.fractal_params)
    xmin, xmax = params["xmin"], params["xmax"]
    ymin, ymax = params["ymin"], params["ymax"]
    
    # 使用向量化计算
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
    z = np.zeros_like(c)
    mask = np.full(c.shape, True, dtype=bool)
    fractal = np.zeros(c.shape)
    
    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask = np.abs(z) < 2.0
        fractal += mask
    
    # 更自然的颜色映射
    fractal = np.sqrt(fractal / max_iter)
    fractal = (fractal * 255).astype(np.uint8)
    
    # 减少模糊，使分形更明显
    fractal = cv2.GaussianBlur(fractal, (5, 5), 1.0)
    
    # 转换为彩色并应用轻微噪声
    color_maps = [cv2.COLORMAP_BONE, cv2.COLORMAP_PINK, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_OCEAN]
    colored_fractal = cv2.applyColorMap(fractal, random.choice(color_maps))
    
    # 增加分形图像的对比度
    colored_fractal = cv2.addWeighted(colored_fractal, 0.9, np.zeros_like(colored_fractal), 0.1, 0)
    
    # 添加更明显的噪声
    noise = np.random.normal(0, 8, colored_fractal.shape).astype(np.uint8)
    colored_fractal = cv2.addWeighted(colored_fractal, 0.92, noise, 0.08, 0)
    
    return colored_fractal

# 创建图像金字塔
def create_pyramid(image, levels):
    pyramid = [image]
    for i in range(1, levels):
        if min(pyramid[-1].shape[:2]) < 32:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid

# 重建金字塔
def reconstruct_pyramid(pyramid):
    reconstructed = pyramid[-1]
    for i in range(len(pyramid)-2, -1, -1):
        expanded = cv2.pyrUp(reconstructed, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        reconstructed = cv2.addWeighted(pyramid[i], 0.5, expanded, 0.5, 0)
    return reconstructed

# 训练掩码生成器
def train_mask_generator(orig_img, gen_img, mask_gen, classifier, clip_processor):
    orig_img = cv2.resize(orig_img, (Config.image_size, Config.image_size))
    gen_img = cv2.resize(gen_img, (Config.image_size, Config.image_size))
    
    orig_tensor = transforms.ToTensor()(orig_img).unsqueeze(0).to(Config.device)
    gen_tensor = transforms.ToTensor()(gen_img).unsqueeze(0).to(Config.device)
    
    inputs = torch.cat([orig_tensor, gen_tensor], dim=1)
    optimizer = optim.Adam(mask_gen.parameters(), lr=Config.mask_lr)
    
    mask_gen.train()
    for epoch in range(Config.mask_train_epochs):
        optimizer.zero_grad()
        mask = mask_gen(inputs)
        mixed_img = (1 - mask) * orig_tensor + mask * gen_tensor
        mixed_pil = transforms.ToPILImage()(mixed_img.squeeze(0).cpu())
        
        try:
            inputs_clip = clip_processor(
                images=[mixed_pil], 
                return_tensors="pt", 
                padding=True
            ).to(Config.device)
            
            with torch.no_grad():
                features = classifier.get_image_features(** inputs_clip)
                confidence = features.norm(dim=1).mean()
        except Exception as e:
            logging.error(f"CLIP处理失败: {str(e)}")
            confidence = torch.tensor(1.0).to(Config.device)
        
        mask_variance = mask.var()
        loss = -confidence + Config.lambda_reg * mask_variance
        loss.backward()
        optimizer.step()
        
        logging.info(f"掩码训练 轮次 {epoch+1}/{Config.mask_train_epochs} - 置信度: {confidence.item():.4f}, 方差: {mask_variance.item():.4f}, 损失: {loss.item():.4f}")
    
    return mask_gen

# 自适应区域保留
def adaptive_mask_blend(orig_img, gen_img, mask_gen, classifier, clip_processor):
    orig_img = cv2.resize(orig_img, (Config.image_size, Config.image_size))
    gen_img = cv2.resize(gen_img, (Config.image_size, Config.image_size))
    
    if min(orig_img.shape[:2]) < Config.min_image_size:
        orig_tensor = transforms.ToTensor()(orig_img).unsqueeze(0).to(Config.device)
        gen_tensor = transforms.ToTensor()(gen_img).unsqueeze(0).to(Config.device)
        inputs = torch.cat([orig_tensor, gen_tensor], dim=1)
        
        with torch.no_grad():
            mask = mask_gen(inputs)
            mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        
        blended = orig_img * (1 - mask) + gen_img * mask
        return blended.astype(np.uint8)
    
    orig_pyramid = create_pyramid(orig_img, Config.pyramid_levels)
    gen_pyramid = create_pyramid(gen_img, Config.pyramid_levels)
    levels = min(len(orig_pyramid), len(gen_pyramid), Config.pyramid_levels)
    blended_pyramid = []
    
    for l in range(levels):
        mask_gen = train_mask_generator(
            orig_pyramid[l], 
            gen_pyramid[l], 
            mask_gen, 
            classifier, 
            clip_processor
        )
        
        orig_tensor = transforms.ToTensor()(orig_pyramid[l]).unsqueeze(0).to(Config.device)
        gen_tensor = transforms.ToTensor()(gen_pyramid[l]).unsqueeze(0).to(Config.device)
        inputs = torch.cat([orig_tensor, gen_tensor], dim=1)
        
        mask_gen.eval()
        with torch.no_grad():
            mask = mask_gen(inputs)
            mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        if mask.shape[:2] != orig_pyramid[l].shape[:2]:
            mask = cv2.resize(mask, (orig_pyramid[l].shape[1], orig_pyramid[l].shape[0]))
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
        
        blended = orig_pyramid[l] * (1 - mask) + gen_pyramid[l] * mask
        blended_pyramid.append(blended.astype(np.uint8))
    
    return reconstruct_pyramid(blended_pyramid)

# 改进的频率域融合 - 增加分形融合效果
def natural_texture_fusion(img1, img2, alpha_range=Config.fractal_alpha_range):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 自适应选择融合系数（增加效果）
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    
    # 转换为YCrCb空间
    ycc1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
    ycc2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    
    # 仅处理Y通道
    y1 = ycc1[:, :, 0].astype(np.float32)
    y2 = ycc2[:, :, 0].astype(np.float32)
    
    # DCT变换
    dct1 = dct(dct(y1.T, norm='ortho').T, norm='ortho')
    dct2 = dct(dct(y2.T, norm='ortho').T, norm='ortho')
    
    # 自适应频率融合 - 主要融合中高频
    h, w = y1.shape
    fused_dct = dct1.copy()
    
    # 低频部分保留原图（减少保留比例）
    low_freq_ratio = 0.2  # 减少低频保留比例
    fused_dct[:int(h*low_freq_ratio), :int(w*low_freq_ratio)] = dct1[:int(h*low_freq_ratio), :int(w*low_freq_ratio)]
    
    # 中频部分混合（增加分形影响）
    mid_freq_ratio = 0.8
    mid_start = int(h*low_freq_ratio)
    mid_end = int(h*mid_freq_ratio)
    fused_dct[mid_start:mid_end, mid_start:mid_end] = (
        (1 - alpha*0.9) * dct1[mid_start:mid_end, mid_start:mid_end] + 
        (alpha*0.9) * dct2[mid_start:mid_end, mid_start:mid_end]
    )
    
    # 高频部分增加混合（增加分形影响）
    high_start = mid_end
    fused_dct[high_start:, high_start:] = (
        (1 - alpha*0.7) * dct1[high_start:, high_start:] + 
        (alpha*0.7) * dct2[high_start:, high_start:]
    )
    
    # IDCT逆变换
    fused_y = idct(idct(fused_dct.T, norm='ortho').T, norm='ortho')
    fused_y = np.clip(fused_y, 0, 255).astype(np.uint8)
    
    # 合并通道
    fused_ycc = ycc1.copy()
    fused_ycc[:, :, 0] = fused_y
    return cv2.cvtColor(fused_ycc, cv2.COLOR_YCrCb2RGB)

# 改进的语义一致性检查
def check_semantic_consistency(orig_img, aug_img, clip_model, clip_processor, orig_label):
    orig_img = Image.fromarray(orig_img).convert("RGB")
    aug_img = Image.fromarray(aug_img).convert("RGB")
    
    try:
        # 使用原始标签
        inputs = clip_processor(
            images=[orig_img, aug_img], 
            text=[orig_label],
            return_tensors="pt", 
            padding=True
        ).to(Config.device)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
        
        # 计算图像-文本相似度
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)
        
        # 原始图像的概率
        orig_prob = probs[0, 0].item()
        # 增强图像的概率
        aug_prob = probs[1, 0].item()
        
        # 宽松的条件：增强图像的概率至少达到原始图像的设定阈值
        consistency = aug_prob >= orig_prob * Config.min_similarity
        
        logging.info(f"语义检查 - 原始: {orig_prob:.4f}, 增强: {aug_prob:.4f}, 阈值: {orig_prob * Config.min_similarity:.4f}, {'通过' if consistency else '失败'}")
        return consistency
    except Exception as e:
        logging.error(f"语义检查失败: {str(e)}")
        return True

# 生成更复杂的随机掩码
def generate_random_mask(height, width):
    """生成更复杂的随机掩码用于图像拼接"""
    mask = np.zeros((height, width), dtype=np.float32)
    
    # 随机选择掩码类型
    mask_type = random.choice(['rectangle', 'ellipse', 'random', 'gradient', 'multiple'])
    
    if mask_type == 'rectangle':
        # 随机矩形
        x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
        x2, y2 = random.randint(width//2, width), random.randint(height//2, height)
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
    elif mask_type == 'ellipse':
        # 随机椭圆
        center_x, center_y = random.randint(width//4, 3*width//4), random.randint(height//4, 3*height//4)
        axes_x, axes_y = random.randint(width//8, width//4), random.randint(height//8, height//4)
        mask = cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 1.0, -1)
    elif mask_type == 'gradient':
        # 渐变掩码
        for i in range(width):
            mask[:, i] = i / width
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
    elif mask_type == 'multiple':
        # 多个形状组合
        for _ in range(3):
            shape_type = random.choice(['rectangle', 'ellipse'])
            if shape_type == 'rectangle':
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(x1, width), random.randint(y1, height)
                mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
            else:
                center_x, center_y = random.randint(0, width), random.randint(0, height)
                axes_x, axes_y = random.randint(width//16, width//8), random.randint(height//16, height//8)
                mask = cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 1.0, -1)
    else:
        # 随机噪声掩码
        mask = np.random.rand(height, width)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = (mask > 0.5).astype(np.float32)
    
    # 应用模糊使边缘更自然
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    return np.expand_dims(mask, axis=2)

# 应用翻转和亮度调整
def apply_flip_and_brightness(image):
    """应用随机翻转和亮度调整"""
    # 随机翻转
    flip_type = random.choice([-1, 0, 1, None])  # -1: 垂直+水平, 0: 垂直, 1: 水平, None: 不翻转
    if flip_type is not None:
        image = cv2.flip(image, flip_type)
    
    # 随机亮度调整
    brightness_factor = random.uniform(0.7, 1.3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return image

# 保存中间结果
def save_debug_image(image, stage, class_name, img_name, attempt=0):
    if not Config.debug_mode:
        return
        
    # 在debug目录下按类别保存
    class_debug_dir = os.path.join(Config.debug_dir, class_name)
    os.makedirs(class_debug_dir, exist_ok=True)
    
    debug_path = os.path.join(class_debug_dir, f"{os.path.splitext(img_name)[0]}_{stage}_att{attempt}.jpg")
    Image.fromarray(image).save(debug_path)
    logging.info(f"保存调试图像: {debug_path}")

# 带重试机制的主增强流程
def frac_diff_aug_with_retry(orig_image, orig_label, pipe, clip_model, clip_processor, mask_gen, class_name, img_name, save_debug):
    for attempt in range(Config.max_retries):
        try:
            result, intermediate_images = frac_diff_aug(orig_image, orig_label, pipe, clip_model, clip_processor, mask_gen, class_name, img_name, attempt, save_debug)
            
            # 如果保存调试信息，则保存中间图像
            if save_debug and intermediate_images:
                for stage, img in intermediate_images.items():
                    save_debug_image(img, stage, class_name, img_name, attempt)
            
            # 最终结果语义检查
            if check_semantic_consistency(orig_image, result, clip_model, clip_processor, orig_label):
                return result
            else:
                logging.warning(f"最终语义检查失败，重试 {attempt+1}/{Config.max_retries}")
        except Exception as e:
            logging.warning(f"增强失败 (尝试 {attempt+1}/{Config.max_retries}): {str(e)}")
            if attempt < Config.max_retries - 1:
                time.sleep(Config.retry_delay)
    
    logging.error("达到最大重试次数，使用原始图像")
    return orig_image

# 添加风格迁移增强（增加效果）
def apply_style_transfer(orig_image, style_image, alpha=0.3):
    """
    应用简单的风格迁移增强（增加效果）
    orig_image: 原始图像 (numpy array)
    style_image: 风格图像 (numpy array)
    alpha: 风格迁移强度 (0.0-1.0)
    """
    # 调整风格图像大小以匹配原始图像
    style_resized = cv2.resize(style_image, (orig_image.shape[1], orig_image.shape[0]))
    
    # 转换为浮点数以便计算
    orig_float = orig_image.astype(np.float32) / 255.0
    style_float = style_resized.astype(np.float32) / 255.0
    
    # 应用简单的颜色迁移
    orig_lab = cv2.cvtColor(orig_float, cv2.COLOR_RGB2LAB)
    style_lab = cv2.cvtColor(style_float, cv2.COLOR_RGB2LAB)
    
    # 迁移亮度和颜色信息
    orig_l, orig_a, orig_b = cv2.split(orig_lab)
    style_l, style_a, style_b = cv2.split(style_lab)
    
    # 混合亮度和颜色通道（增加效果）
    mixed_l = (1 - alpha) * orig_l + alpha * style_l
    mixed_a = (1 - alpha) * orig_a + alpha * style_a
    mixed_b = (1 - alpha) * orig_b + alpha * style_b
    
    # 合并通道并转换回RGB
    mixed_lab = cv2.merge([mixed_l, mixed_a, mixed_b])
    mixed_rgb = cv2.cvtColor(mixed_lab, cv2.COLOR_LAB2RGB)
    
    # 限制值在0-1之间并转换回uint8
    mixed_rgb = np.clip(mixed_rgb * 255, 0, 255).astype(np.uint8)
    
    return mixed_rgb

# 主增强流程 - 调整后的流程：扩散 → 分形融合 → 掩码拼接
def frac_diff_aug(orig_image, orig_label, pipe, clip_model, clip_processor, mask_gen, class_name, img_name, attempt=0, save_debug=True):
    orig_pil = Image.fromarray(orig_image)
    w, h = orig_pil.size
    
    intermediate_images = {}  # 保存中间图像用于调试
    
    # 保存原始图像到中间结果（如果保存调试信息）
    if save_debug:
        intermediate_images["original"] = orig_image
    
    # 1. 使用扩散模型生成扩散图像 - 增加扩散强度，产生更明显的变化
    prompt = generate_labeled_prompt(orig_label)
    logging.info(f"使用提示词: {prompt}")
    
    # 使用较高的扩散强度，产生更明显的变化
    strength = random.uniform(Config.min_strength, Config.max_strength)
    
    generator = torch.Generator(device=Config.device).manual_seed(random.randint(0, 1000000))
    try:
        # 使用中等推理步骤
        generated = pipe(
            prompt=prompt,
            image=orig_pil,
            strength=strength,
            guidance_scale=7.5,  # 中等引导比例
            num_inference_steps=40,  # 中等推理步骤
            generator=generator
        ).images[0]
    except Exception as e:
        logging.error(f"扩散模型生成失败: {str(e)}")
        generated = orig_pil
    
    # 调整大小并转换格式
    diffused_img = np.array(generated.resize((w, h)))
    
    # 保存扩散图像到中间结果
    if save_debug:
        intermediate_images["diffused"] = diffused_img
    
    # 2. 扩散图像语义检查
    if not check_semantic_consistency(orig_image, diffused_img, clip_model, clip_processor, orig_label):
        logging.warning("扩散生成图像语义不一致，使用原始图像")
        diffused_img = orig_image.copy()
    
    # 3. 生成分形图像
    fractal_img = generate_natural_fractal(h, w)
    
    # 保存分形图像到中间结果
    if save_debug:
        intermediate_images["fractal"] = fractal_img
    
    # 4. 将分形图像按一定比例融入扩散图像生成融合图像 - 增加分形融合比例
    fused_img = natural_texture_fusion(diffused_img, fractal_img)
    
    # 保存融合图像到中间结果
    if save_debug:
        intermediate_images["fused"] = fused_img
    
    # 5. 生成随机掩码并在保留关键语义的基础上将扩散图像和融合图像拼接
    mask = generate_random_mask(h, w)
    final_img = (diffused_img * (1 - mask) + fused_img * mask).astype(np.uint8)
    
    # 保存最终图像到中间结果
    if save_debug:
        intermediate_images["final"] = final_img
    
    # 6. 最终融合结果检查
    final_consistency = check_semantic_consistency(orig_image, final_img, clip_model, clip_processor, orig_label)
    logging.info(f"最终语义一致性: {'通过' if final_consistency else '失败'}")
    
    return final_img, intermediate_images

# 批量处理函数
def process_dataset():
    try:
        pipe, clip_model, clip_processor, mask_gen = init_models()
    except Exception as e:
        logging.error(f"模型初始化失败: {str(e)}")
        return
    
    for class_name in Config.classes:
        orig_dir = os.path.join(Config.data_dir, class_name)
        result_dir = os.path.join(Config.result_dir, class_name)
        
        logging.info(f"处理类别 {class_name}: {orig_dir} -> {result_dir}")
        orig_label = f"class_{class_name}"
        
        # 获取所有图像文件并筛选图片
        all_images = [
            img_name for img_name in os.listdir(orig_dir)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # 随机选择要保存调试信息的图像（每个类别5张）
        num_debug = min(Config.debug_samples_per_class, len(all_images))
        debug_images = set(random.sample(all_images, num_debug))
        logging.info(f"类别 {class_name} 将为 {num_debug} 张图像保存调试信息")
        
        # 准备保存所有图像（原始、最终、翻转）
        all_images_to_save = []
        
        # 处理所有图像
        for img_idx, img_name in enumerate(all_images):
            img_path = os.path.join(orig_dir, img_name)
            try:
                # 判断是否需要保存调试信息
                save_debug = img_name in debug_images
                
                orig_img = np.array(Image.open(img_path).convert("RGB"))
                
                # 应用增强流程
                enhanced_img = frac_diff_aug_with_retry(
                    orig_img, 
                    orig_label,
                    pipe, 
                    clip_model, 
                    clip_processor,
                    mask_gen,
                    class_name,
                    img_name,
                    save_debug
                )
                
                # 保存原始图像和增强图像
                all_images_to_save.append(("original", orig_img))
                all_images_to_save.append(("enhanced", enhanced_img))
                
                # 为原始图像和增强图像生成翻转版本
                flipped_orig = apply_flip_and_brightness(orig_img)
                flipped_enhanced = apply_flip_and_brightness(enhanced_img)
                
                all_images_to_save.append(("flipped_original", flipped_orig))
                all_images_to_save.append(("flipped_enhanced", flipped_enhanced))
                
                logging.info(f"处理完成: {img_name}")
                
            except Exception as e:
                logging.error(f"处理 {img_path} 时出错: {str(e)}")
        
        # 将所有图像保存并重命名为1-200
        logging.info(f"开始保存类别 {class_name} 的图像，共 {len(all_images_to_save)} 张")
        
        for idx, (img_type, img) in enumerate(all_images_to_save):
            if idx < 200:  # 确保不超过200张
                img_num = idx + 1
                result_path = os.path.join(result_dir, f"{img_num}.jpg")
                Image.fromarray(img).save(result_path)
        
        logging.info(f"类别 {class_name} 图像保存完成，共保存 {min(len(all_images_to_save), 200)} 张图像")

if __name__ == "__main__":
    logging.info("开始数据增强处理...")
    process_dataset()
    logging.info("数据增强完成!")