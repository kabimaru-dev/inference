# 1. CREATE AND ACTIVATE A VIRTUAL ENVIRONMENT
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. INSTALL DEPENDENCIES
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sentencepiece

# 3. DOWNLOAD MODEL WEIGHTS (Llama-3.1-8B-Instruct in GGUF format)
# This command downloads a specific quantization of the model.
# You can change 'Q4_K_M' to another quant level (e.g., Q5_K_M, Q8_0).
# See https://huggingface.co/TheBloke for other models/quantizations.
# --progress-bar is for a better download experience.
huggingface-cli download TheBloke/Meta-Llama-3.1-8B-Instruct-GGUF --local-dir Meta-Llama-3.1-8B-Instruct-GGUF --local-dir-use-symlink False --progress-bar bar

# The actual model file will be in a subfolder like 'Meta-Llama-3.1-8B-Instruct-GGUF/...'
# Find the .gguf file, it will be named something like 'meta-llama-3.1-8b-instruct.Q4_K_M.gguf'
# Let's assume the file is named 'meta-llama-3.1-8b-instruct.Q4_K_M.gguf' for the script below.

# 4. PYTHON INFERENCE SCRIPT (save as `inference.py`)
cat << 'EOF' > inference.py
import os
from pathlib import Path

# --- 1. SETUP ---
# IMPORTANT: Replace with the actual path to your downloaded .gguf file
# If you are in the same directory as the downloaded 'Meta-Llama-3.1-8B-Instruct-GGUF' folder:
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct-GGUF/meta-llama-3.1-8b-instruct.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please download the model and update the MODEL_PATH variable.")
    exit()

# Ensure the model file ends with .gguf
if not MODEL_PATH.lower().endswith(".gguf"):
    raise ValueError("MODEL_PATH must point to a .gguf file.")

# --- 2. LOAD MODEL AND TOKENIZER ---
# Uses the 'auto' backend to pick the fastest available (e.g., llama-cpp-python or ctransformers)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Set to True if using a 4-bit quant model
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading model and tokenizer... (this may take a while)")
model = AutoModelForCausalLM.from_pretrained(
    Path(MODEL_PATH).parent, # Point to the directory containing the .gguf file
    device_map="auto",
    torch_dtype="auto",
    quantization_config=bnb_config if MODEL_PATH.lower().endswith('.gguf') else None,
    trust_remote_code=True,
    # The following line is crucial for GGUF models in some setups
    use_safetensors=False 
)
tokenizer = AutoTokenizer.from_pretrained(Path(MODEL_PATH).parent, trust_remote_code=True)
print("Model and tokenizer loaded successfully.")


# --- 3. INFERENCE FUNCTION ---
def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """
    Generates a response from the model based on a given prompt.
    """
    print(f"\n[USER] {prompt}")
    print("[ASSISTANT] ", end="", flush=True)

    # Format the prompt for the model
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Generate the response
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode and print the response
    response_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    print(response_text)

# --- 4. RUN INFERENCE ---
if __name__ == "__main__":
    print("\n--- Starting Inference Loop ---")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            generate_response(model, tokenizer, user_input)

        except KeyboardInterrupt:
            print("\n\nInference interrupted by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

    print("\n--- Inference Ended ---")
    del model
    del tokenizer
EOF

# 5. RUN THE INFERENCE SCRIPT
# This command will execute the Python script you just created.
python inference.py