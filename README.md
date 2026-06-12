# YOLO-Training-Visualizer

<div align="center">

### **A real-time, interactive logging board tailored for parsing YOLO training logs (`results.csv`) and rendering advanced computer vision metrics.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green.svg?style=for-the-badge&logo=target&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

⭐ **Star this repo if it helps you!** ⭐

🔥 **Share it with the community!** 🔥

[![Share on X](https://img.shields.io/badge/Share_on-X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20Image%20Classification%20project!%20🤖✨%20https://github.com/Mushrum-mmb/Simple-AI-Image-Classification%20%23AI%20%23MachineLearning%20%23DeepLearning)
[![Share on Facebook](https://img.shields.io/badge/Share_on-Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/Mushrum-mmb/Simple-AI-Image-Classification)
[![Share on LinkedIn](https://img.shields.io/badge/Share_on-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/Mushrum-mmb/Simple-AI-Image-Classification)
[![Share on Reddit](https://img.shields.io/badge/Share_on-Reddit-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/submit?title=Amazing%20AI%20Image%20Classification%20Project&url=https://github.com/Mushrum-mmb/Simple-AI-Image-Classification)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Gallery](#-gallery)
- [Local Usage](#%EF%B8%8F-local-usage)
- [Google Colab Usage](#-google-colab-usage)
- [How It Works](#-how-it-works)
- [Contributing](#-contributing)
- [License](#-license)


## 🚀 About

<div align="center">

**YOLO Training Visualizer** is a lightweight, intuitive visualization tool built on top of **Gradio**. This application allows computer vision engineers and developers to seamlessly upload their `results.csv` log files (generated during Ultralytics YOLO model training) to analyze, compare, and monitor essential performance metrics interactively in real time.

</div>

<div align="center">

| **Interface** | **Framework** | **Author** |
|:---:|:---:|:---:|
| Web_based | Gradio | [Mushrum-mmb](https://github.com/Mushrum-mmb/) |

</div>

### 🌟 **Key Highlights:**
- **Flexible File Upload**: Supports dragging and dropping or directly browsing to upload your `results.csv` file.
- **Real-time Graphs**: Automatically parses data and renders responsive, interactive charts.
- **Comprehensive Metric Tracking**:
  - **Loss Metrics**: Box Loss, Cls Loss, DFL Loss (for both Train and Validation sets).
  - **Performance Metrics**: Precision (B), Recall (B), mAP50 (B), mAP50-95 (B).
- **User-friendly Interface**: Clean, modern UI with built-in support for Gradio's automatic light/dark mode toggling.
---

## 📸 Gallery

<div align="center">
    
### 🔍 **See the Magic in Action!**

*Examples of successful classify training visualization*
<div align="center">

<img width="1170" height="864" alt="image" src="https://github.com/user-attachments/assets/acd32af6-79a9-4569-87a5-29aee7a8ad18" />

</div>

*Detection Training*
<div align="center">
<img width="1160" height="757" alt="image" src="https://github.com/user-attachments/assets/1164290b-8008-45a5-998a-bef7d5b1bd10" />

</div>

*Pose Estimation*

<div align="center">

<img width="1164" height="766" alt="image" src="https://github.com/user-attachments/assets/330c3f6a-92ad-44a1-9dad-3a946e42318a" />
   
</div>

*Segmentation*

</div>
<img width="1162" height="760" alt="image" src="https://github.com/user-attachments/assets/2936f6d9-2413-4c99-af2c-a914deb4bba1" />

</div>

---

## ✨ Features

<div align="center">

### **What Makes This Special?**

</div>

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Real-time Parsing** | Extracts live training metrics from Ultralytics `results.csv` files | Instant insights without reading raw data |
| **Comprehensive Charts** | Visualizes both complex Loss functions and Performance metrics | Clear oversight of model convergence |
| **Instant Inference** | Dynamic graph rendering immediately upon file upload | No lag, smooth user experience |
| **Gradio Web UI** |Modern, reactive web interface accessible directly via browser | No complex GUI dependencies required |
| **Easy Deployment** | Launches with a single command and supports public link sharing | Seamless collaboration and quick previews |
| **Zero Setup Overheads** | Lightweight implementation running purely on the CPU/GPU local backend | Works on any device without heavy deep learning environments |

<div align="center">

### 🎯 **Perfect For:**
**Students** • **Researchers** • **Developers** 

</div>

---

## ▶️ Local Usage

<div align="center">

### 🚀 **Launch this App in 4 Simple Steps!**

</div>

**Step 1:** Clone the repository
```bash
git clone https://github.com/Mushrum-mmb/YOLO-Training-Visualizer.git
```

**Step 2:** Navigate to project directory
```bash
cd YOLO-Training-Visualizer
```

**Step 3:** Install the requirements
```bash
pip install -r requirements.txt
```

**Step 4:** Launch the application
```bash
python app.py
```

<div align="center">
    
*🎉 Your App is Ready! Open the provided link in your browser and start visualizing trainings!*

</div>

<img width="866" height="288" alt="image" src="https://github.com/user-attachments/assets/499c5f8d-a5b4-41f4-a284-fdb041f85f1d" />


---

## 💻 Google Colab Usage

<div align="center">

### ☁️ **Perfect for Potato Computers!** 🥔

[![Open In Colab](https://img.shields.io/badge/Open_in-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1BF0_3p7gzR7V3YiGQX97JrS4q769pErK?usp=sharing)

</div>

Can't run AI on your device? No problem! Use our optimized Google Colab notebook for seamless cloud-based AI training and inference.

<details>
<summary>📖 <strong>Colab Guide (Click to expand)</strong></summary>

*Just execute the first and second cell*

<img width="1792" height="749" alt="image" src="https://github.com/user-attachments/assets/0384ad75-f37a-4443-8047-456f6befcc78" />


**Launch and enjoy! 🎉**

</details>

---

## 🔧 How It Works

<div align="center">

### **Architecture Overview**

</div>

Our real-time logging board processes CSV telemetry and hardware data through a centralized pipeline:

<div align="center">

```mermaid
graph TD
    A[results.csv Input] -->|Pandas Parsing| B{Task Matcher}
    B -->|'train/seg_loss'| C[segmentation_task]
    B -->|'train/pose_loss'| D[pose_estimation_task]
    B -->|'metrics/accuracy_top1'| E[classification_task]
    B -->|'train/box_loss'| F[detection_task]
    
    G[psutil & pynvml] -->|Hardware Telemetry| H[System Resource Plots]
    
    C & D & E & F & H -->|Matplotlib Figures| I[Gradio Timer Event]
    I -->|Reactive Rendering| J[Gradio Interface Dashboard]
```

</div>

| Component | Purpose | Key Features |
|-----------|---------|-------------|
| **Task Matcher** | Log Classification | • Automatically detects YOLO task types from CSV headers.<br>• Routes logs to matching analysis tracks. |
| **Task Functions** | Metric Processing | • Evaluates training/validation metrics and learning rates.<br>• Computes best scores using a colored threshold picker. |
| **Resource Monitors** | Hardware Telemetry | • Tracks rolling CPU, RAM, and Disk metrics via `psutil`.<br>• Hooks into NVIDIA VRAM logs via `pynvml`. |
| **App Interface** | Dashboard Loop | • Drives seamless page-switching inside a terminal theme.<br>• Uses `gr.Timer` to stream data updates without reloading. |

---

## 🤝 Contributing

<div align="center">

### 💡 **Help Make This Project Even Better!**

[![Contributors Welcome](https://img.shields.io/badge/Contributors-Welcome-brightgreen?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Mushrum-mmb/Simple-AI-Image-Classification/issues)

</div>

We love contributions from the community! Here's how you can help:

- **Report bugs** or suggest features
- **Submit pull requests** with improvements
- **Improve documentation** and tutorials
- **Share your results** and use cases
- **Star the repo** to show support!

---

## 📜 License

<div align="center">

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

</div>
