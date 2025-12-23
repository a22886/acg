## Environment Requirements

Make sure your python version is around 3.10 for taichi support.
```
pip install \
trimesh pysplashsurf taichi bpy tqdm scipy \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
--extra-index-url https://download.blender.org/pypi

sudo apt install ffmpeg
```

## Run

Set up your scene and configurations in `main.py` and run `./render.sh`. If you don't change anything, the output video will be fluid falling with no rigid objects, although the rigid objects are in fact defined.

## Videos

https://github.com/user-attachments/assets/aaf33575-ef38-4f19-9e69-c99cbe07bf5c

https://github.com/user-attachments/assets/180383f0-e2aa-4614-b1b9-397b08ed0c17