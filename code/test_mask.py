from PIL import Image

# 读取原始灰度图像
for i in range(65,69):
    original_image = Image.open(r"G:\小论文\消融实验跑图用\#7\model\result\{}\img.png".format(i)).convert("L")
    mask = Image.open(r"G:\小论文\消融实验跑图用\#5\model\pred\patient0{}_frame01_pred.nii\slice_000.png".format(i)).convert("L")

    # 创建一个空白的RGB图像，与原始图像的大小一致
    colored_image = Image.new("RGB", original_image.size)

    # 定义颜色映射
    color_map = {
        85: (255, 0, 0),    # RV的颜色为红色
        170: (0, 255, 0),    # LV的颜色为绿色
        255: (0, 0, 255),    # MYO的颜色为蓝色
    }

    # 遍历每个像素，根据mask中的像素值为每个像素赋予颜色
    for label, color in color_map.items():
        mask_pixels = mask.point(lambda p: p == label and 255)
        colored_image.paste(Image.new("RGB", original_image.size, color), mask=mask_pixels)

    # 使用原始图像覆盖mask中像素值为0的部分
    colored_image = Image.composite(colored_image, Image.new("RGB", original_image.size), mask)

    # 显示原始图像
    original_image.show()

    # 显示分割效果
    colored_image.show()

    #保存分割效果
    save_path = r"G:\小论文\消融实验跑图用\#5\model\result\{}\seg.png".format(i)
    colored_image.save(save_path)











