# Car Detection Project

This project contains code and dataset files for vehicle detection and color recognition.

## What is included

- `car_detection/` - detector code and project files
- `CarD_1.ipynb` - notebook for experiments or training
- `requirements.txt` - Python dependencies
- `yolov8n.pt` - YOLOv8 model weights
- `car_color_model.h5` - color recognition model weights
- `Intersection Traffic Dataset.v1-v1.0.yolov8/` - dataset split for detection
- `VCoR (Vehicle Color Recognition) Dataset/` - color recognition dataset

## Recommended GitHub setup

1. Create a GitHub repository, e.g. `car_detection_project`
2. Make sure `.gitignore` includes:
   - `venv/`
   - `__pycache__/`
   - `*.pyc`, `*.pyo`, `*.pyd`
   - `*.h5`, `*.pt`
   - `.ipynb_checkpoints/`
   - `.gradio/`
   - generated output images like `output_result.jpg`
3. If not already initialized, run:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   ```
4. Add the GitHub remote and push:
   ```powershell
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```

## Recommended Git settings on Windows

To avoid line-ending warnings, set:
```powershell
git config core.autocrlf false
```

Also keep `.gitattributes` in the repo with:
```text
* text=auto
*.txt text eol=lf
*.py text eol=lf
*.md text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
```

## Recreate the Python environment

Run:
```powershell
python -m pip install -r requirements.txt
```

## Notes

- Do not push `venv/` to GitHub.
- Do not push large model weight files unless you intend to store them in Git LFS.
- If you want the repository to remain small, keep model files outside the repo or add them via Git LFS.
