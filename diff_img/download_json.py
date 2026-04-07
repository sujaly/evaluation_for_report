import os
import requests
from pathlib import Path

def download_json_with_resume():
    """
    下载JSON文件，支持断点续传
    源路径: /mnt/mimic-cxr/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph/0a27a7a6-c3bb9cfa-956e0eef-2c19e165-1687ea63_SceneGraph.json
    目标路径: D:\pyworkspace\evaluation_for_report\data\sceneGraph_json
    """
    # 源文件路径 (使用file://协议进行本地文件复制)
    source_path = "/mnt/mimic-cxr/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph/0a27a7a6-c3bb9cfa-956e0eef-2c19e165-1687ea63_SceneGraph.json"
    
    # 目标文件夹
    target_dir = r"D:\pyworkspace\evaluation_for_report\data\sceneGraph_json"
    
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 目标文件路径
    filename = os.path.basename(source_path)
    target_path = os.path.join(target_dir, filename)
    temp_path = target_path + ".tmp"
    
    try:
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            print(f"错误: 源文件不存在 - {source_path}")
            return False
        
        # 获取源文件大小
        source_size = os.path.getsize(source_path)
        print(f"源文件大小: {source_size} 字节")
        
        # 检查是否已存在完整下载的文件
        if os.path.exists(target_path):
            target_size = os.path.getsize(target_path)
            if target_size == source_size:
                print(f"文件已完整下载，跳过下载")
                print(f"文件路径: {target_path}")
                return True
            else:
                print(f"文件不完整，重新下载 (已下载: {target_size}/{source_size} 字节)")
                os.remove(target_path)
        
        # 检查临时文件是否存在（断点续传）
        downloaded_size = 0
        if os.path.exists(temp_path):
            downloaded_size = os.path.getsize(temp_path)
            print(f"发现未完成的下载，已下载: {downloaded_size} 字节")
            
            if downloaded_size >= source_size:
                # 临时文件已经完整，直接重命名
                os.rename(temp_path, target_path)
                print(f"文件下载完成！")
                print(f"文件路径: {target_path}")
                return True
        
        # 使用二进制模式复制文件，支持断点续传
        chunk_size = 8192  # 8KB 块大小
        
        # 以追加模式打开临时文件
        mode = 'ab' if downloaded_size > 0 else 'wb'
        
        with open(source_path, 'rb') as src_file:
            # 如果已下载部分，跳过已下载的内容
            if downloaded_size > 0:
                src_file.seek(downloaded_size)
                print(f"从 {downloaded_size} 字节处继续下载...")
            
            with open(temp_path, mode) as dst_file:
                copied = downloaded_size
                while True:
                    chunk = src_file.read(chunk_size)
                    if not chunk:
                        break
                    
                    dst_file.write(chunk)
                    copied += len(chunk)
                    
                    # 显示进度
                    progress = (copied / source_size) * 100
                    print(f"\r下载进度: {progress:.1f}% ({copied}/{source_size} 字节)", end='', flush=True)
        
        print()  # 换行
        
        # 下载完成，重命名临时文件
        os.rename(temp_path, target_path)
        
        # 验证文件
        if os.path.exists(target_path):
            final_size = os.path.getsize(target_path)
            print(f"文件下载成功！")
            print(f"文件路径: {target_path}")
            print(f"文件大小: {final_size} 字节")
            
            if final_size == source_size:
                print("文件完整性验证通过")
                return True
            else:
                print(f"警告: 文件大小不匹配 (期望: {source_size}, 实际: {final_size})")
                return False
        else:
            print("错误: 文件重命名失败")
            return False
            
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return False
    except PermissionError as e:
        print(f"错误: 权限不足 - {e}")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def verify_download(target_path, expected_size=None):
    """验证下载的文件"""
    if not os.path.exists(target_path):
        print(f"文件不存在: {target_path}")
        return False
    
    actual_size = os.path.getsize(target_path)
    print(f"文件存在，大小: {actual_size} 字节")
    
    if expected_size and actual_size != expected_size:
        print(f"文件大小不匹配 (期望: {expected_size}, 实际: {actual_size})")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("JSON文件下载工具 (支持断点续传)")
    print("=" * 60)
    
    success = download_json_with_resume()
    
    print("=" * 60)
    if success:
        print("下载完成！")
    else:
        print("下载失败！")
    print("=" * 60)
