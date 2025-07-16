import gradio as gr
import os, shutil

os.makedirs("data", exist_ok=True)

user_selections = {
    "model": "Qwen2.5-0.5B Instruct",
    "finetune_method": "Full Finetuning",
    "algorithm": "Supervised Finetuning",
    "uploaded_file": None
}

def handle_upload(file):
    destination = os.path.join("data", os.path.basename(file.name))
    shutil.copy(file.name, destination)
    user_selections["uploaded_file"] = destination
    return f"Uploaded: {destination}"

def update_model_choice(choice):
    user_selections["model"] = choice

def update_finetune_method(choice):
    user_selections["finetune_method"] = choice

def update_algorithm(choice):
    user_selections["algorithm"] = choice

def start_training():
    file = user_selections["uploaded_file"]
    model = user_selections["model"]
    method = user_selections["finetune_method"]
    algo = user_selections["algorithm"]

    if not file:
        return "Please upload a dataset before training."

    return f" Received the following configuration for fine-tuning: \n\n File: {file}\n Model: {model}\n Method: {method}\n Algorithm: {algo}"

with gr.Blocks() as demo:
    gr.HTML("<center><h1>FinetuneX: Tune LLMs Your Way</h1></center>")
    gr.Markdown("<center><h2> Upload your dataset and get a fine-tuned LLM that understands your needs.. </h2></center>")

    with gr.Row():
        with gr.Column():
            status = gr.Textbox(label="Status", value="Upload a dataset file (.csv or .json) to get started.", interactive=False)
            upload_button = gr.UploadButton("Upload", file_types=[".csv", ".json"], file_count="single")
            upload_button.upload(handle_upload, upload_button, status)

    with gr.Row():
        with gr.Column():
            model_choice = gr.Dropdown(
                choices=["Qwen2.5-0.5B Instruct", "LLaMA 3.1 (Coming Soon)", "Mistral 7B (Coming Soon)", "Gemma 2 (Coming Soon)"],
                value="Qwen2.5-0.5B Instruct",
                label="Model"
            )
            model_choice.change(update_model_choice, model_choice)

        with gr.Column():
            finetune_method = gr.Dropdown(
                choices=["Full Finetuning", "LoRA (Coming Soon)", "QLoRA (Coming Soon)"],
                value="Full Finetuning",
                label="Fine-Tuning Type"
            )
            finetune_method.change(update_finetune_method, finetune_method)

        with gr.Column():
            algo = gr.Dropdown(
                choices=["Supervised Finetuning", "Direct Preference Optimization (Coming Soon)", "Proximal Policy Optimization (Coming Soon)", "REINFORCE with a leave-one-out (Coming Soon)"],
                value="Supervised Finetuning",
                label="Training Algorithm"
            )
            algo.change(update_algorithm, algo)

    with gr.Row():
        with gr.Column():
            start_button = gr.Button("Start Training!")
            training_status = gr.Textbox(label="Training Status", interactive=False)
            start_button.click(start_training, None, training_status)

demo.launch()