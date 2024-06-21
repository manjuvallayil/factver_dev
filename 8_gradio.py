## test samples
# Peter Liese is a german lawmaker
# Sony and Honda to begin delivering vehicles in 2026
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the base model and tokenizer
base_model_name = "meta-llama/Llama-2-7b-chat-hf" #meta-llama/Llama-2-7b-chat-hf
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the fine-tuned model and tokenizer
fine_tuned_model_name = "manjuvallayil/Llama-2-7b-chat-finetune-factver"
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_name)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_name)

# Setup pipelines
base_pipe = pipeline(task="text-generation", model=base_model, tokenizer=base_tokenizer, max_length=200)
fine_tuned_pipe = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, max_length=200)


def generate_response(claim):
    print('Claim is: ', claim)
    prompt = f"<s>[INST] {claim}, You are an Automated fact verification assistant. You are supposed to classify a Claim into one of the following label categories (T for True, F for False and N for Not Enough Info) and you also have to generate an Evidence as an explanation, keep the answer as concise as possible. If a claim does not make any sense, or is not factually coherent, explain why instead of answering something not correct.[/INST]"
    print('Prompt is: ', prompt)
    base_result = base_pipe(claim)[0]['generated_text']
    if base_result.startswith(prompt): # Clean the response to ensure it doesn't include the prompt
        base_result = base_result[len(prompt):].strip()
    fine_tuned_result = fine_tuned_pipe(prompt)[0]['generated_text']
    if fine_tuned_result.startswith(prompt): # Clean the response to ensure it doesn't include the prompt
        fine_tuned_result = fine_tuned_result[len(prompt):].strip()
    return base_result, fine_tuned_result

# Setup Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a claim here..."),
    ],
    outputs=[
        gr.Textbox(label="Response from Base Model"),
        gr.Textbox(label="Response from Fine-Tuned Model")
    ],
    title="Fact Verification with LLaMA Models",
    description="Enter a claim to see how the base and fine-tuned LLaMA models respond. The models generate an explanation based on the claim."
)

iface.launch()
