NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from . import nodes_gguf
    
    NODE_CLASS_MAPPINGS = {
        "JJC_JoyCaption_GGUF": nodes_gguf.JoyCaptionGGUF,
        "JJC_JoyCaption_Custom_GGUF": nodes_gguf.JoyCaptionCustomGGUF,
        "JJC_JoyCaption_GGUF_ExtraOptions": nodes_gguf.JoyCaptionGGUFExtraOptions,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "JJC_JoyCaption_GGUF": "JoyCaption (GGUF)",
        "JJC_JoyCaption_Custom_GGUF": "JoyCaption (Custom GGUF)",
        "JJC_JoyCaption_GGUF_ExtraOptions": "JoyCaption GGUF Extra Options",
    }
    print("[JoyCaption] GGUF nodes loaded.")

except ImportError:
    print("[JoyCaption] GGUF nodes not available. 'llama-cpp-python' 未安装或安装不正确。")
    print("[JoyCaption] 正在尝试自动安装 llama-cpp-python...")
    
    try:
        import subprocess
        import sys
        import os
        import platform
        
        # 检测CUDA版本
        try:
            import torch
            cuda_version = torch.version.cuda
            if cuda_version:
                if cuda_version.startswith("11"):
                    cuda_tag = "cu118"
                elif cuda_version.startswith("12"):
                    cuda_tag = "cu121" # 12.1兼容性最广泛
                else:
                    cuda_tag = "cu121" # 默认使用cu121
            else:
                cuda_tag = "cu121" # 如果检测不到CUDA版本，默认使用cu121
            print(f"[JoyCaption] 检测到CUDA版本: {cuda_version if cuda_version else '未检测到'}, 使用标签: {cuda_tag}")
        except:
            cuda_tag = "cu121"
            print(f"[JoyCaption] 无法检测CUDA版本，使用默认标签: {cuda_tag}")
        
        # 设置环境变量
        env = os.environ.copy()
        
        # 根据系统设置编译参数
        if platform.system() == "Windows":
            # Windows可能需要不同的编译器设置
            env["CMAKE_ARGS"] = "-DLLAMA_OPENMP=ON"
        else:
            # Linux/Docker环境
            env["CMAKE_ARGS"] = "-DLLAMA_OPENMP=ON -DLLAMA_CUBLAS=ON"
            # 检查编译器是否可用
            gcc_available = subprocess.run(["which", "gcc"], capture_output=True, text=True).returncode == 0
            gpp_available = subprocess.run(["which", "g++"], capture_output=True, text=True).returncode == 0
            
            if gcc_available and gpp_available:
                env["CMAKE_ARGS"] += " -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++"
                print("[JoyCaption] 使用GCC/G++编译器")
            else:
                print("[JoyCaption] 未找到GCC/G++编译器，使用系统默认编译器")
        
        # 设置安装命令
        install_cmd = [
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "llama-cpp-python==0.3.4", 
            "--force-reinstall", 
            "--no-cache-dir", 
            "--extra-index-url", 
            f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag}"
        ]
        
        print(f"[JoyCaption] 执行安装命令: {' '.join(install_cmd)}")
        print(f"[JoyCaption] 使用编译选项: {env['CMAKE_ARGS']}")
        
        # 执行安装命令
        result = subprocess.run(install_cmd, env=env, capture_output=True, text=True)
        
        # 输出安装结果
        if result.returncode == 0:
            print("[JoyCaption] llama-cpp-python 安装成功！重新尝试加载节点...")
            # 安装成功后尝试再次导入
            from . import nodes_gguf
            
            NODE_CLASS_MAPPINGS = {
                "JJC_JoyCaption_GGUF": nodes_gguf.JoyCaptionGGUF,
                "JJC_JoyCaption_Custom_GGUF": nodes_gguf.JoyCaptionCustomGGUF,
                "JJC_JoyCaption_GGUF_ExtraOptions": nodes_gguf.JoyCaptionGGUFExtraOptions,
            }
            NODE_DISPLAY_NAME_MAPPINGS = {
                "JJC_JoyCaption_GGUF": "JoyCaption (GGUF)",
                "JJC_JoyCaption_Custom_GGUF": "JoyCaption (Custom GGUF)",
                "JJC_JoyCaption_GGUF_ExtraOptions": "JoyCaption GGUF Extra Options",
            }
            print("[JoyCaption] GGUF节点加载成功!")
        else:
            print(f"[JoyCaption] llama-cpp-python 安装失败：\n{result.stderr}")
            print("[JoyCaption] 请手动安装 llama-cpp-python，使用以下命令：")
            print(f"CMAKE_ARGS=\"{env['CMAKE_ARGS']}\" pip install llama-cpp-python==0.3.4 --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag}")
    except Exception as e:
        print(f"[JoyCaption] 自动安装过程中发生错误：{e}")
        print("[JoyCaption] 请手动安装 llama-cpp-python：")
        print("CMAKE_ARGS=\"-DLLAMA_OPENMP=ON -DLLAMA_CUBLAS=ON\" pip install llama-cpp-python==0.3.4 --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
    # Mappings will remain empty if both import and installation fail
except Exception as e:
    print(f"[JoyCaption] Error loading GGUF nodes: {e}")
    # Mappings will remain empty if import fails

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
