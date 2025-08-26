import gradio as gr
import os, shutil
from main import main
from inference import infer, infer_base
import traceback

os.makedirs("data", exist_ok=True)

user_selections = {
    "model": "Qwen2.5-0.5B",
    "finetune_method": "Full Finetuning",
    "algorithm": "Supervised Finetuning",
    "uploaded_file": None,
    "training_complete": False,
    "finetuned_model_path": None,
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
    
    initial_message = f"""
    Training Started!
    
    File: {file}
    Model: {model}
    Method: {method}
    Algorithm: {algo}
    
    Please wait while training is in progress...
    """
    
    try:
        main(file, model)  # This is where the actual training happens
        
        user_selections["finetuned_model_path"] = "./finetuned"
        user_selections["training_complete"] = True
        
        final_message = f"""
        Training Complete! 
        
        File: {file}
        Model: {model}
        Method: {method}
        Algorithm: {algo}
        
        Your model has been successfully fine-tuned!
        Check out the Chat and Battle Arena tabs!
        """
        return final_message
        
    except Exception as e:
        user_selections["training_complete"] = False
        full_traceback = traceback.format_exc()
        error_message = f"""
        Training Failed! 
        
        Error: {str(e)}
        StackTrace: {full_traceback}
        Please check your dataset and try again.
        """
        return error_message

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<center><h2>FinetuneX: Tune LLMs Your Way</h2></center>")
    gr.Markdown("<center><h3> Upload your dataset and get a fine-tuned LLM that understands your needs.. </h3></center>")
    
    with gr.Tab("Train & Monitor"):
        with gr.Row():
            with gr.Column():
                status = gr.Textbox(label="Status", value="Upload a dataset file (.csv or .json) to get started.", interactive=False)
                upload_button = gr.UploadButton("Upload", file_types=[".csv", ".json"], file_count="single")
                upload_button.upload(handle_upload, upload_button, status)

        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=["Qwen2.5-0.5B", "LLaMA 3.1 (Coming Soon)", "Mistral 7B (Coming Soon)", "Gemma 2 (Coming Soon)"],
                    value="Qwen2.5-0.5B",
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
                
                def training_with_updates():
                    result = start_training()
                    
                    if user_selections["training_complete"]:
                        return (
                            result,  
                            gr.update(value="Model Ready!"),
                            gr.update(interactive=True, placeholder="Ask your fine-tuned model anything..."),  
                            gr.update(interactive=True),  
                            gr.update(value="Models ready for comparison!"), 
                            gr.update(interactive=True, placeholder="Ask both models the same question to compare..."), 
                            gr.update(interactive=True)  
                        )
                    else:
                        return (
                            result, 
                            gr.update(value="Training incomplete. Please try again."), 
                            gr.update(interactive=False),  
                            gr.update(interactive=False),  
                            gr.update(value="Training incomplete. Please complete training first."), 
                            gr.update(interactive=False),  
                            gr.update(interactive=False)  
                        )
    
    with gr.Tab("Chat"):
        with gr.Column():
            model_status = gr.Textbox(
                label="Model Status",
                value="Please complete training first to use the chat interface.",
                interactive=False
            )
            with gr.Row():
                gr.Markdown("### Chat with your fine-tuned model")
            
            chatbot = gr.Chatbot(
                label="Your Fine-tuned Model",
                height=400,
                placeholder="Complete training first to start chatting with your fine-tuned model..."
            )
            
            msg = gr.Textbox(
                label="Type your message",
                placeholder="Ask your fine-tuned model anything...",
                interactive=False
            )
            send_button = gr.Button("Send", interactive=False)
            
            def chat_with_model(history, user_message):
                if not user_message or not user_selections["training_complete"]:
                    return history, ""
                
                model_path = user_selections["finetuned_model_path"]
                selected_model = user_selections["model"]
                response = f"Response from fine-tuned model: {infer(user_message, model_path, selected_model)}"  
                history.append((user_message, response))
                return history, ""
            
            send_button.click(
                chat_with_model,
                inputs=[chatbot, msg],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                chat_with_model,
                inputs=[chatbot, msg],
                outputs=[chatbot, msg]
            )

    with gr.Tab("Battle Arena"):  
        with gr.Column():
            comparison_status = gr.Textbox(
                label="Comparison Status",
                value="Please complete training first to compare models.",
                interactive=False
            )  
            gr.Markdown("### Utilize Dual Chat Mode")
            gr.Markdown("Compare the difference between base model and your finetuned model")
            with gr.Row():
                with gr.Column():
                    base_chatbot = gr.Chatbot(
                        label="Base Model Response",
                        height=350,
                        placeholder="Base model responses will appear here..."
                    )
                
                with gr.Column():
                    finetuned_chatbot = gr.Chatbot(
                        label="Your Fine-tuned Model Response", 
                        height=350,
                        placeholder="Fine-tuned model responses will appear here..."
                    )
            
            compare_msg = gr.Textbox(
                label="Message to both models",
                placeholder="Ask both models the same question to compare...",
                interactive=False
            )
                   
            compare_button = gr.Button("Compare", interactive=False)
            
            def compare_models(history_base, history_finetune, prompt):
                if not prompt or not user_selections["training_complete"]:
                    return history_base, history_finetune, ""
            
                base_model_path = user_selections["model"]
                finetuned_model_path = user_selections["finetuned_model_path"]
            
                base_resp = f"Base model response: {infer_base(prompt, base_model_path)}" 
                finetuned_resp = f"Fine-tuned model response: {infer(prompt, finetuned_model_path)}"  
            
                history_base.append((prompt, base_resp))
                history_finetune.append((prompt, finetuned_resp))
                return history_base, history_finetune, ""
            
            compare_button.click(
                compare_models,
                inputs=[base_chatbot, finetuned_chatbot, compare_msg],
                outputs=[base_chatbot, finetuned_chatbot, compare_msg]
            )
            
            compare_msg.submit(
                compare_models,
                inputs=[base_chatbot, finetuned_chatbot, compare_msg],
                outputs=[base_chatbot, finetuned_chatbot, compare_msg]
            )

    start_button.click(
        training_with_updates,
        inputs=None,
        outputs=[training_status, model_status, msg, send_button, comparison_status, compare_msg, compare_button]
    )

demo.launch()