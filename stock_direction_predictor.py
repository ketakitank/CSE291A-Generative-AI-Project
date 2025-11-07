import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from rich.console import Console
from rich.table import Table
from rich.text import Text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import re
import numpy as np

# --- Configuration ---
# Paste your Hugging Face token here if you use a gated model like Llama 3
# Not needed for Qwen 1.5
HF_TOKEN = ""  
MODEL_NAME = "Qwen/qwen1.5-1.8b-chat"  # A great, small, non-gated model
NEW_MODEL_NAME = "qwen-1.8b-financial-predictor"  # Local name for your fine-tuned model
DATA_PATH = "merged.csv"

# --- Rich Console Setup ---
console = Console()

def print_rich(text, style="bold green"):
    """Prints rich formatted text to the console."""
    console.print(Text(text, style=style))

def create_finetuning_dataset(data_path: str, test_size: float = 0.1) -> (Dataset, Dataset, pd.DataFrame):
    """
    Loads, processes, and formats the dataset for fine-tuning.

    Handles:
    - Loading the CSV
    - Filling rows with no news
    - Sorting by ticker and date to build time-series
    - Creating the 'label' (UP/DOWN) by shifting
    - Formatting into the LLM prompt structure
    - Splitting into train and test sets
    """
    print_rich(f"Loading data from {data_path}...", style="cyan")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print_rich(f"Error: The file {data_path} was not found.", style="bold red")
        return None, None, None

    # --- Data Cleaning and Feature Engineering ---
    
    # We need prices to determine the label. Drop rows where prices are missing.
    df = df.dropna(subset=['prices'])
    
    # *** LOGIC UPDATED AS PER YOUR REQUEST ***
    # Instead of dropping rows with no news, fill them with an explicit signal.
    df['news'] = df['news'].fillna("No specific news reported for this day.")

    # Convert date to datetime objects for correct sorting
    df['date'] = pd.to_datetime(df['date'])

    # IMPORTANT: Sort by ticker and date to ensure time-series integrity
    df = df.sort_values(by=['ticker', 'date'])

    # --- Label Creation ---
    
    # 1. Get the last price for the current day (t)
    # The 'prices' column is a string "p1,p2,...,p15". We split by ',' and take the last one.
    def get_last_price(price_str):
        try:
            return float(str(price_str).split(',')[-1])
        except (ValueError, IndexError):
            return None
            
    df['price_t'] = df['prices'].apply(get_last_price)
    
    # 2. Get the last price for the *next* day (t+1)
    # We use groupby('ticker') to ensure we're not mixing tickers.
    # .shift(-1) looks one row *forward* in time.
    df['price_t1'] = df.groupby('ticker')['price_t'].shift(-1)

    # 3. Drop rows where we can't create a label
    # This removes the last day of each ticker (no t+1 to compare to)
    # and any rows where price parsing failed.
    df = df.dropna(subset=['price_t', 'price_t1'])

    # 4. Create the classification label
    df['label'] = np.where(df['price_t1'] > df['price_t'], 'UP', 'DOWN')

    print_rich(f"Processed {len(df)} datapoints.", style="cyan")
    
    # Keep a copy of the test dataframe for later evaluation
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
    
    # --- Prompt Formatting ---
    def format_prompt(row):
        """
        Formats a single row of data into the [INST]...[/INST] format.
        """
        # Clean the news text to remove artifacts (optional, but good practice)
        news_cleaned = re.sub(r'[\r\n]+', ' ', str(row['news']))
        news_cleaned = re.sub(r'\s{2,}', ' ', news_cleaned).strip()

        prompt = f"""<s>[INST]
Analyze the following financial data for ticker {row['ticker']} and predict the stock price direction for the next trading day.

### Ticker:
{row['ticker']}

### Price History (last 15 days):
{row['prices']}

### Recent News:
{news_cleaned}

Based on this data, will the price move UP or DOWN?
[/INST] {row['label']} </s>"""
        return prompt

    print_rich("Formatting prompts...", style="cyan")
    
    train_prompts = [format_prompt(row) for _, row in train_df.iterrows()]
    test_prompts = [format_prompt(row) for _, row in test_df.iterrows()]
    
    # Convert to Hugging Face Dataset object
    train_ds = Dataset.from_dict({"text": train_prompts})
    test_ds = Dataset.from_dict({"text": test_prompts})

    return train_ds, test_ds, test_df

def train_model(train_ds: Dataset):
    """
    Loads a base model, configures it for 4-bit QLoRA, and runs the fine-tuning.
    """
    
    if HF_TOKEN:
        print_rich("Logging into Hugging Face Hub...", style="yellow")
        try:
            login(token=HF_TOKEN)
            print_rich("Login successful.", style="green")
        except Exception as e:
            print_rich(f"Login failed: {e}. Check your HF_TOKEN.", style="bold red")
            return None
    else:
        print_rich("No HF_TOKEN provided. Proceeding (this will fail for gated models).", style="yellow")

    # --- Model Loading Configuration ---

    # 4-bit quantization for memory saving
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load the base model
    print_rich(f"Loading base model: {MODEL_NAME}", style="cyan")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically uses the GPU
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- PEFT (LoRA) Configuration ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen 1.5 specific target modules
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    print_rich("PEFT model configured.", style="cyan")

    # --- Training Configuration ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,  # Start with 1 epoch, increase to 3 if needed
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        optim="paged_adamw_32bit",
        logging_steps=25,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        evaluation_strategy="no", # No eval set during training for speed
        report_to="none",
    )

    # --- Trainer Initialization ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    # --- Start Training ---
    print_rich("--- Starting Model Training ---", style="bold magenta")
    trainer.train()
    print_rich("--- Model Training Complete ---", style="bold magenta")

    # --- Save the Model ---
    print_rich(f"Saving fine-tuned model to {NEW_MODEL_NAME}", style="cyan")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    
    return model, tokenizer

def evaluate_model(test_df: pd.DataFrame, model, tokenizer):
    """
    Evaluates the fine-tuned model on the hold-out test set.
    """
    print_rich("\n--- Starting Model Evaluation ---", style="bold magenta")
    
    predictions = []
    ground_truth = []

    for _, row in test_df.iterrows():
        # Format the input prompt (without the answer)
        news_cleaned = re.sub(r'[\r\n]+', ' ', str(row['news']))
        news_cleaned = re.sub(r'\s{2,}', ' ', news_cleaned).strip()
        
        prompt_text = f"""<s>[INST]
Analyze the following financial data for ticker {row['ticker']} and predict the stock price direction for the next trading day.

### Ticker:
{row['ticker']}

### Price History (last 15 days):
{row['prices']}

### Recent News:
{news_cleaned}

Based on this data, will the price move UP or DOWN?
[/INST]"""

        # Tokenize the input
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        
        # Generate the output
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5,  # We only need 'UP' or 'DOWN'
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        # The output includes the prompt, so we skip the prompt tokens
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Simple parsing of the response
        if "UP" in response_text:
            pred = "UP"
        elif "DOWN" in response_text:
            pred = "DOWN"
        else:
            pred = "UNKNOWN" # Model failed to answer clearly

        predictions.append(pred)
        ground_truth.append(row['label'])
        
        # Optional: Log progress
        if (len(predictions) > 0 and len(predictions) % 50 == 0):
            print(f"Evaluated {len(predictions)} / {len(test_df)} examples...")

    # --- Display Results ---
    print_rich("\n--- Evaluation Results ---", style="bold magenta")
    
    accuracy = accuracy_score(ground_truth, predictions)
    print_rich(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(ground_truth, predictions, labels=["UP", "DOWN", "UNKNOWN"])
    table = Table(title="Confusion Matrix")
    table.add_column("Actual", justify="right", style="cyan")
    table.add_column("Pred UP", justify="right", style="green")
    table.add_column("Pred DOWN", justify="right", style="red")
    table.add_column("Pred UNKNOWN", justify="right", style="yellow")
    
    # Add rows to table
    labels = ["UP", "DOWN", "UNKNOWN"]
    for i, label in enumerate(labels):
        if label == "UNKNOWN" and sum(cm[i]) == 0 and "UNKNOWN" not in ground_truth: continue # Skip if no UNKNOWN
        table.add_row(
            label,
            str(cm[i, 0]),
            str(cm[i, 1]),
            str(cm[i, 2])
        )
    console.print(table)

    # Classification Report
    print_rich("\nClassification Report:", style="bold white")
    # Filter out 'UNKNOWN' if it has no support, as it causes warnings
    report_labels = ["UP", "DOWN"]
    if "UNKNOWN" in predictions or "UNKNOWN" in ground_truth:
        report_labels.append("UNKNOWN")
        
    print(classification_report(ground_truth, predictions, labels=report_labels, zero_division=0))


# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Create the dataset
    train_ds, test_ds, test_dataframe = create_finetuning_dataset(DATA_PATH)
    
    if train_ds is None:
        print_rich("Dataset creation failed. Exiting.", style="bold red")
        exit()

    # 2. Train the model
    model, tokenizer = train_model(train_ds)

    # 3. Evaluate the model
    if model and tokenizer and test_dataframe is not None:
        evaluate_model(test_dataframe, model, tokenizer)
    else:
        print_rich("Training failed. Skipping evaluation.", style="bold red")