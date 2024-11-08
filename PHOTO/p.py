from PIL import Image
import os

# 開啟原始圖片
for i in range(360):
    image = Image.open(r'C:\Users\a2778\GitHub\FanAOI\FanAOI\PHOTO\0.png')

    # 旋轉角度 (度數)
    angle = i  # 設定你想要的角度
    rotated_image = image.rotate(angle, expand=True)  # 旋轉圖片並保持圖片大小

    # 設定新的儲存路徑
    save_dir = r'C:\Users\a2778\GitHub\FanAOI\FanAOI\PHOTO'  # 設定儲存資料夾
    os.makedirs(save_dir, exist_ok=True)  # 如果資料夾不存在，則創建它

    # 設定檔案名稱
    save_path = os.path.join(save_dir, str(i)+'.png')

    # 保存結果
    rotated_image.save(save_path)

    print(f"圖片已保存為: {save_path}")
