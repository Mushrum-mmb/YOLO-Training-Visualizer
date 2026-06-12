import gradio as gr
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import os
import pynvml

# Initialize GPU if available
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

# History buffers
ram_hist, cpu_hist,  gpu_hist, disk_hist = [], [], [], []

# --- RAM ---
def plot_ram():
    global ram_hist

    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024**3)
    ram_total = ram.total / (1024**3)
    ram_hist.append(ram_used)
    if len(ram_hist) > 50: ram_hist.pop(0)

    fig = plt.figure(figsize=(3,2))
    plt.style.use("dark_background")
    plt.plot(ram_hist, color="green")
    plt.title(f"System RAM ({ram_used:.1f}/{ram_total:.1f} GB)")
    plt.xlabel("Time")
    plt.ylabel("GB")
    plt.ylim(0, ram_total)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    return fig

# --- CPU ---
def plot_cpu():
    global cpu_hist

    cpu = psutil.cpu_percent(interval=None)  # % usage
    cpu_hist.append(cpu)
    if len(cpu_hist) > 50: cpu_hist.pop(0)
    fig = plt.figure(figsize=(3,2))
    plt.style.use("dark_background")
    plt.plot(cpu_hist, color="red")
    plt.title(f"CPU Usage ({cpu:.1f}%)")
    plt.xlabel("Time"); plt.ylabel("%")
    plt.ylim(0, 100)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    return fig

# --- GPU ---
def plot_gpu():
    global gpu_hist

    gpu_used, gpu_total = 0, 1
    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used = meminfo.used / (1024**3)
        gpu_total = meminfo.total / (1024**3)
    gpu_hist.append(gpu_used)
    if len(gpu_hist) > 50: gpu_hist.pop(0)

    fig = plt.figure(figsize=(3,2))
    plt.style.use("dark_background")
    if GPU_AVAILABLE:
        plt.plot(gpu_hist, color="orange")
        plt.title(f"GPU RAM ({gpu_used:.1f}/{gpu_total:.1f} GB)")
        plt.ylim(0, gpu_total)
    else:
        plt.text(0.5, 0.5, "GPU Not Available", ha="center", va="center")
        plt.title("GPU RAM")
    plt.xlabel("Time")
    plt.ylabel("GB")
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    return fig

# --- Disk ---
def plot_disk():
    global disk_hist

    disk = psutil.disk_usage('/')
    disk_used = disk.used / (1024**3)
    disk_total = disk.total / (1024**3)
    disk_hist.append(disk_used)
    if len(disk_hist) > 50: disk_hist.pop(0)

    fig = plt.figure(figsize=(3,2))
    plt.style.use("dark_background")
    plt.plot(disk_hist, color="blue")
    plt.title(f"Disk ({disk_used:.1f}/{disk_total:.1f} GB)")
    plt.xlabel("Time")
    plt.ylabel("GB")
    plt.ylim(0, disk_total)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    return fig

#--- Color picker ---
def color_score_picker(sc, thr1, thr2, thr3, loss):
    if loss:
        if sc > thr1:
            color = "#7CFC00"
        elif sc > thr2 and sc < thr1:
            color = "#2E8B57"
        elif sc > thr3 and sc < thr2:
            color = "#DAA520"
        elif sc < thr3:
            color = "#800000"
    else:
        if sc < thr1:
            color = "#7CFC00"
        elif sc > thr1 and sc < thr2:
            color = "#2E8B57"
        elif sc > thr2 and sc < thr3:
            color = "#DAA520"
        elif sc > thr3:
            color = "#800000"
    return color

# --- Detection task ---
def plot_losses_train_d(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid()

    return fig

def plot_metric_val_d(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="precision(B)")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="recall(B)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid(True)
    return fig

def plot_losses_val_d(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
    plt.plot(df["epoch"], df["val/dfl_loss"], label="val/dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.legend()
    plt.grid(True)
    return fig

# --- Classification task ---
def plot_losses_train_c(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["train/loss"], label="train/loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    return fig

def plot_metric_val_c(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="acc_top1")
    plt.plot(df["epoch"], df["metrics/accuracy_top5"], label="acc_top5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy")
    plt.legend()
    plt.grid(True)
    return fig

def plot_losses_val_c(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["val/loss"], label="val/loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.legend()
    plt.grid(True)
    return fig

# --- Segmentation task ---
def plot_losses_train_s(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
    plt.plot(df["epoch"], df["train/seg_loss"], label="train/seg_loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    return fig

def plot_metric_val_s(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50(B)")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95(B)")
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="precision(B)")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="recall(B)")
    plt.plot(df["epoch"], df["metrics/mAP50(M)"], label="mAP50(M)")
    plt.plot(df["epoch"], df["metrics/mAP50-95(M)"], label="mAP50-95(M)")
    plt.plot(df["epoch"], df["metrics/precision(M)"], label="precision(M)")
    plt.plot(df["epoch"], df["metrics/recall(M)"], label="recall(M)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy")
    plt.legend()
    plt.grid(True)
    return fig

def plot_losses_val_s(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
    plt.plot(df["epoch"], df["val/dfl_loss"], label="val/dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.legend()
    plt.grid(True)
    return fig

# --- Pose estimation task ---
def plot_losses_train_p(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
    plt.plot(df["epoch"], df["train/pose_loss"], label="train/pose_loss")
    plt.plot(df["epoch"], df["train/kobj_loss"], label="train/kobj_loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    return fig

def plot_metric_val_p(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50(B)")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95(B)")
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="precision(B)")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="recall(B)")
    plt.plot(df["epoch"], df["metrics/mAP50(P)"], label="mAP50(P)")
    plt.plot(df["epoch"], df["metrics/mAP50-95(P)"], label="mAP50-95(P)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy")
    plt.legend()
    plt.grid(True)
    return fig

def plot_losses_val_p(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
    plt.plot(df["epoch"], df["val/pose_loss"], label="val/pose_loss")
    plt.plot(df["epoch"], df["val/kobj_loss"], label="val/kobj_loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
    plt.plot(df["epoch"], df["val/dfl_loss"], label="val/dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.legend()
    plt.grid(True)
    return fig

def plot_lr(df):
    fig = plt.figure(figsize=(10,6))
    plt.style.use("dark_background")
    plt.plot(df["epoch"], df["lr/pg0"], label="pg0")
    plt.plot(df["epoch"], df["lr/pg1"], label="pg1")
    plt.plot(df["epoch"], df["lr/pg2"], label="pg2")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate")
    plt.legend()
    plt.grid(True)
    return fig

def detection_task(df):
    fig_loss_train = plot_losses_train_d(df)
    fig_loss_val = plot_losses_val_d(df)
    fig_metric = plot_metric_val_d(df)
    fig_lr = plot_lr(df)
    fig_ram = plot_ram()
    fig_cpu = plot_cpu()
    fig_gpu = plot_gpu()
    fig_disk = plot_disk()
    plt.close(fig_loss_train)
    plt.close(fig_loss_val)
    plt.close(fig_metric)
    plt.close(fig_lr)
    plt.close(fig_ram)
    plt.close(fig_cpu)
    plt.close(fig_gpu)
    plt.close(fig_disk)
    info_epoch = f"""
    <p style="background-color:#3B82F6 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
        Lasted Epoch: {len(df)}
    </p>
    """
    info_lr = f"""
        <p style="background-color:#10B981 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Lasted Learning Rate: {df['lr/pg0'].iloc[-1]}
        </p>
    """
    info_m1 = f"""
        <p style="background-color:{color_score_picker(df['metrics/mAP50(B)'].max(), 0.85, 0.75, 0.6, True)} !important; 
        font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Best Metrics/mAP50(B) Score: {df['metrics/mAP50(B)'].max()}
        </p>
    """
    info_m2 = f"""
        <p style="background-color:{color_score_picker(df['metrics/mAP50-95(B)'].max(), 0.85, 0.75, 0.6, True)} !important;
        font-family: 'Courier New', monospace; 
        color:#ffffff !important; 
        padding:10px;
        ">
            Best Metrics/mAP50-95(B) Score: {df['metrics/mAP50-95(B)'].max()}
        </p>
    """
    return (fig_loss_train,
            fig_loss_val,
            fig_metric,
            fig_lr,
            fig_ram,
            fig_cpu,
            fig_gpu,
            fig_disk,
            info_epoch,
            info_lr,
            info_m1,
            info_m2
    )

def classification_task(df):
    fig_loss_train = plot_losses_train_c(df)
    fig_loss_val = plot_losses_val_c(df)
    fig_metric = plot_metric_val_c(df)
    fig_lr = plot_lr(df)
    fig_ram = plot_ram()
    fig_cpu = plot_cpu()
    fig_gpu = plot_gpu()
    fig_disk = plot_disk()
    plt.close(fig_loss_train)
    plt.close(fig_loss_val)
    plt.close(fig_metric)
    plt.close(fig_lr)
    plt.close(fig_ram)
    plt.close(fig_cpu)
    plt.close(fig_gpu)
    plt.close(fig_disk)
    info_epoch = f"""
    <p style="background-color:#3B82F6 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
        Lasted Epoch: {len(df)}
    </p>
    """
    info_lr = f"""
        <p style="background-color:#10B981 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Lasted Learning Rate: {df['lr/pg0'].iloc[-1]}
        </p>
    """
    info_m1 = f"""
        <p style="background-color:{color_score_picker(df['metrics/accuracy_top1'].max(), 0.85, 0.75, 0.6, True)} !important; 
        font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Best Accuracy_top1 Score: {df['metrics/accuracy_top1'].max()}
        </p>
    """
    info_m2 = f"""
        <p style="background-color:{color_score_picker(df['metrics/accuracy_top5'].max(), 0.85, 0.75, 0.6, True)} !important;
        font-family: 'Courier New', monospace; 
        color:#ffffff !important; 
        padding:10px;
        ">
            Best Accuracy_top5 Score: {df['metrics/accuracy_top5'].max()}
        </p>
    """
    return (fig_loss_train,
            fig_loss_val,
            fig_metric,
            fig_lr,
            fig_ram,
            fig_cpu,
            fig_gpu,
            fig_disk,
            info_epoch,
            info_lr,
            info_m1,
            info_m2
    )

def segmentation_task(df):
    fig_loss_train = plot_losses_train_s(df)
    fig_loss_val = plot_losses_val_s(df)
    fig_metric = plot_metric_val_s(df)
    fig_lr = plot_lr(df)
    fig_ram = plot_ram()
    fig_cpu = plot_cpu()
    fig_gpu = plot_gpu()
    fig_disk = plot_disk()
    plt.close(fig_loss_train)
    plt.close(fig_loss_val)
    plt.close(fig_metric)
    plt.close(fig_lr)
    plt.close(fig_ram)
    plt.close(fig_cpu)
    plt.close(fig_gpu)
    plt.close(fig_disk)
    info_epoch = f"""
    <p style="background-color:#3B82F6 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
        Lasted Epoch: {len(df)}
    </p>
    """
    info_lr = f"""
        <p style="background-color:#10B981 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Lasted Learning Rate: {df['lr/pg0'].iloc[-1]}
        </p>
    """
    info_m1 = f"""
        <p style="background-color:{color_score_picker(df['metrics/mAP50(B)'].max(), 0.85, 0.75, 0.6, True)} !important; 
        font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Best Metrics/mAP50(B) Score: {df['metrics/mAP50(B)'].max()}
        </p>
    """
    info_m2 = f"""
        <p style="background-color:{color_score_picker(df['metrics/mAP50(M)'].max(), 0.85, 0.75, 0.6, True)} !important;
        font-family: 'Courier New', monospace; 
        color:#ffffff !important; 
        padding:10px;
        ">
            Best Metrics/mAP50(M) Score: {df['metrics/mAP50(M)'].max()}
        </p>
    """
    return (fig_loss_train,
            fig_loss_val,
            fig_metric,
            fig_lr,
            fig_ram,
            fig_cpu,
            fig_gpu,
            fig_disk,
            info_epoch,
            info_lr,
            info_m1,
            info_m2
    )

def pose_estimation_task(df):
    fig_loss_train = plot_losses_train_p(df)
    fig_loss_val = plot_losses_val_p(df)
    fig_metric = plot_metric_val_p(df)
    fig_lr = plot_lr(df)
    fig_ram = plot_ram()
    fig_cpu = plot_cpu()
    fig_gpu = plot_gpu()
    fig_disk = plot_disk()
    plt.close(fig_loss_train)
    plt.close(fig_loss_val)
    plt.close(fig_metric)
    plt.close(fig_lr)
    plt.close(fig_ram)
    plt.close(fig_cpu)
    plt.close(fig_gpu)
    plt.close(fig_disk)
    info_epoch = f"""
    <p style="background-color:#3B82F6 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
        Lasted Epoch: {len(df)}
    </p>
    """
    info_lr = f"""
        <p style="background-color:#10B981 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Lasted Learning Rate: {df['lr/pg0'].iloc[-1]}
        </p>
    """
    info_m1 = f"""
        <p style="background-color:{color_score_picker(df['metrics/mAP50(B)'].max(), 0.85, 0.75, 0.6, True)} !important; 
        font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;">
            Best Metrics/mAP50(B) Score: {df['metrics/mAP50(B)'].max()}
        </p>
    """
    info_m2 = f"""
        <p style="background-color:{color_score_picker(df['metrics/mAP50(P)'].max(), 0.85, 0.75, 0.6, True)} !important;
        font-family: 'Courier New', monospace; 
        color:#ffffff !important; 
        padding:10px;
        ">
            Best Metrics/mAP50(P) Score: {df['metrics/mAP50(P)'].max()}
        </p>
    """
    return (fig_loss_train,
            fig_loss_val,
            fig_metric,
            fig_lr,
            fig_ram,
            fig_cpu,
            fig_gpu,
            fig_disk,
            info_epoch,
            info_lr,
            info_m1,
            info_m2
    )

def create_chart(result_path):
    if not os.path.exists(result_path):
        return (None, None, None, None, None, None, None, None,
        "⚠️ No results.csv found yet. Train first!",
        "⚠️ No results.csv found yet. Train first!",
        "⚠️ No results.csv found yet. Train first!",
        "⚠️ No results.csv found yet. Train first!")
    task = "unknown"
    df = pd.read_csv(result_path)
    columns = set(df.columns)

    if 'train/seg_loss' in columns:
        (losses_train_plot,
        losses_val_plot,
        metrics_plot,
        lr_plot,
        ram_plot,
        cpu_plot,
        gpu_plot,
        disk_plot,
        info_epoch,
        info_lr,
        info_m1,
        info_m2) = segmentation_task(df)
    elif 'train/pose_loss' in columns:
        (losses_train_plot,
        losses_val_plot,
        metrics_plot,
        lr_plot,
        ram_plot,
        cpu_plot,
        gpu_plot,
        disk_plot,
        info_epoch,
        info_lr,
        info_m1,
        info_m2) = pose_estimation_task(df)
    elif 'metrics/accuracy_top1' in columns:
        (losses_train_plot,
        losses_val_plot,
        metrics_plot,
        lr_plot,
        ram_plot,
        cpu_plot,
        gpu_plot,
        disk_plot,
        info_epoch,
        info_lr,
        info_m1,
        info_m2) = classification_task(df)
    elif 'train/box_loss' in columns:
        (losses_train_plot,
        losses_val_plot,
        metrics_plot,
        lr_plot,
        ram_plot,
        cpu_plot,
        gpu_plot,
        disk_plot,
        info_epoch,
        info_lr,
        info_m1,
        info_m2) = detection_task(df)
    return (losses_train_plot,
            losses_val_plot,
            metrics_plot,
            lr_plot,
            ram_plot,
            cpu_plot,
            gpu_plot,
            disk_plot,
            info_epoch,
            info_lr,
            info_m1,
            info_m2
    )

# --- HTML Info ---
footer = """
<div style="text-align: center; font-family: 'Courier New', monospace; margin-top: 20px;">
    <a href="https://www.linkedin.com/in/namush-bui-555948335/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/Mushrum-mmb" target="_blank">GitHub</a> |
    <a href="https://linktr.ee/Namush" target="_blank">Linktree</a><br>
    Made with 💖 by Namush
</div>
"""

def switch_page(show_dashboard):
    return (
        gr.update(visible= not show_dashboard),
        gr.update(visible=show_dashboard)
    )
def change_time_update(time):
    return gr.update(value=time)

def turn_auto_update(active):
    new_state = not active
    btn_label = "⏸ Stop Auto Update" if new_state else "▶ Auto Update"
    return new_state, gr.update(active=new_state), gr.update(value=btn_label)


# --- Build Gradio UI ---
with gr.Blocks(css="""
    #btn_continue { color: #000000; }
    #settings_tab { color: #ffffff; }

    /* Tab buttons */
    button[role="tab"] {
        background-color: #046b2f !important;
        color: #ffffff!important;
        border: 1px solid #ffffff !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Tab hover */
    button[role="tab"]:hover {
        background-color: #22c00e !important;
        color: #ffffff !important;
    }

    /* Tab selected */
    button[role="tab"].selected {
        background-color: #046b2f !important;
        color: #ffffff !important;
    }
    /* Markdown text */
    .markdown p, .markdown span, .prose p {
        color: #ffffff !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Info boxes (background colored ones) */
    .markdown p strong {
        color: #ffffff !important;
    }
""",
    theme=gr.Theme.from_hub("hmb/terminal"), js="""
    function() {
        document.documentElement.classList.add('dark');
    }
""") as app:
    gr.Markdown("""
        <div style="text-align: center; font-family: 'Courier New', monospace; font-size: 24px; font-weight: bold;">
        YOLO Training Visualizer
        </div>
    """)
    dash_board_visible = gr.State(False)
    auto_state = gr.State(True)

    with gr.Group(visible=True) as file_page:
        with gr.Row():
            result_path = gr.Textbox(label="Paste your results.csv file path here", value=r"C:\Users\DELL\Downloads\YOLO-Training-Visualizer\tests\results_detection.csv")
        with gr.Row():
            btn_continue = gr.Button("Continue", elem_id="btn_continue")
    with gr.Group(visible=False) as dash_board_page:
        gr.Markdown("""
                <div style="text-align: left; font-family: 'Courier New', monospace; font-size: 16px; font-weight: bold;">
                System Info:
                </div>
        """)
        with gr.Row():
            with gr.Column():
                ram_plot = gr.Plot(show_label=False)
            with gr.Column():
                cpu_plot = gr.Plot(show_label=False)
            with gr.Column():
                gpu_plot = gr.Plot(show_label=False)
            with gr.Column():
                disk_plot = gr.Plot(show_label=False)
        with gr.Tab("Result Board", elem_id="settings_tab"):
            gr.Markdown("""
                <div style="text-align: left; font-family: 'Courier New', monospace; font-size: 16px; font-weight: bold;">
                Performance Info:
                </div>
            """)
            with gr.Row():
                with gr.Column():
                    info_epoch = gr.Markdown(
                        "<p style='background-color:#3B82F6 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;'>Lasted Epoch: Loading...</p>")
                with gr.Column():
                    info_lr = gr.Markdown(
                        "<p style='background-color:#10B981 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;'>Lasted Learning Rate: Loading...</p>")
            with gr.Row():
                with gr.Column():
                    info_m1 = gr.Markdown(
                        "<p style='background-color:#F59E0B !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;'>Score: Loading...</p>")
                with gr.Column():
                    info_m2 = gr.Markdown(
                        "<p style='background-color:#EF4444 !important; font-family: 'Courier New', monospace; color:#ffffff !important; padding:10px;'>Score: Loading...</p>")
            with gr.Row():
                with gr.Column():
                    losses_train_plot = gr.Plot(label="Training Losses")
                with gr.Column():
                    losses_val_plot = gr.Plot(label="Validation Losses")
            with gr.Row():
                with gr.Column():
                    metrics_plot = gr.Plot(label="Validation metrics")
                with gr.Column():
                    lr_plot = gr.Plot(label="Learning Rate")
        with gr.Tab("🛠 Settings", elem_id="settings_tab"):
            with gr.Row():
                with gr.Column():
                    time_slider = gr.Slider(minimum=1, maximum=151, step=15, value=15, label="Time auto-update")
                with gr.Column():
                    auto_btn = gr.Button("▶ Auto-update", elem_id="btn_continue")


        with gr.Row():
            btn_back =gr.Button("Choose another path", elem_id="btn_continue")
    with gr.Row():
        gr.HTML(footer)


    btn_continue.click(fn=lambda: True, inputs=None, outputs=dash_board_visible)
    btn_continue.click(
        switch_page,
        inputs=[dash_board_visible],
        outputs=[file_page, dash_board_page],
    )

    btn_back.click(fn=lambda: False, inputs=None, outputs=dash_board_visible)
    btn_back.click(
        switch_page,
        inputs=[dash_board_visible],
        outputs=[file_page, dash_board_page],
    )

    # --- Timers ---
    chart_timer = gr.Timer(value=10, active=True)
    chart_timer.tick(fn=create_chart, inputs=result_path,
                     outputs=[
                          losses_train_plot,
                          losses_val_plot,
                          metrics_plot,
                          lr_plot,
                          ram_plot,
                          cpu_plot,
                          gpu_plot,
                          disk_plot,
                          info_epoch,
                          info_lr,
                          info_m1,
                          info_m2
                    ])

    time_slider.change(
        fn=change_time_update,
        inputs=time_slider,
        outputs=chart_timer
    )

    auto_btn.click(
        fn=turn_auto_update,
        inputs=auto_state,
        outputs=[auto_state, chart_timer, auto_btn]
    )

if __name__ == '__main__':
    app.launch(share= True, show_error=True)