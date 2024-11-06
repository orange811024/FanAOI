from PIL import Image
import os
for i in range(360):
    # 開啟原始圖片
    image = Image.open(r'C:\Users\a2778\GitHub\pythontvdi\xan\X1.jpg')

    # 開啟圓形圖檔
    circle_image = Image.open(r'C:\Users\a2778\GitHub\pythontvdi\xan\X3.png')

    # 設定圓形的中心位置
    circle_center = (175, 170)  # 圓心座標

    # 旋轉圓形圖檔
    angle = i  # 旋轉角度 (度數)
    rotated_circle = circle_image.rotate(angle)  # 旋轉圓形並保持圖片大小expand=True

    # 計算圓形旋轉後的位置，將其貼回原圖
    paste_position = (circle_center[0] - rotated_circle.width // 2, 
                    circle_center[1] - rotated_circle.height // 2)

    # 將旋轉後的圓形圖檔貼回到原圖中
    image.paste(rotated_circle, paste_position, rotated_circle)  # 使用圓形的 alpha 通道作為遮罩

    # 顯示結果
    #image.show()

    # 設定新的儲存路徑
    save_dir =(r'C:\Users\a2778\GitHub\pythontvdi\xan')  # 設定儲存資料夾
    os.makedirs(save_dir, exist_ok=True)  # 如果資料夾不存在，則創建它

    # 檢查流水號
    file_index = 1
    while os.path.exists(os.path.join(save_dir, f'modified_image_{file_index}.png')):
        file_index += 1

    # 最終檔案名稱
    save_path = os.path.join(save_dir, f'modified_image_{file_index}.png')

    # 保存結果
    image.save(save_path)

    print(f"圖片已保存為: {save_path}")
