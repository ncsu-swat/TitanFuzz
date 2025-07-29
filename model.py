# model.py
import re
import time
import os
import ollama


class LanguageModel:
    """
    A unified interface for generating code from a language model, supporting
    both local Ollama models and legacy Hugging Face models.
    """
    def __init__(self, model_identifier: str, num_samples: int = 1):
        print(f"Initializing Language Model: {model_identifier} ...")
        t_start = time.time()
        self.model_identifier = model_identifier
        self.num_samples = num_samples # Default samples per call
        
        # This is the generic placeholder for modern instruction-tuned models.
        self.infill_ph = "<FILL_ME>"

        if model_identifier.startswith("ollama/"):
            self.backend = "ollama"
            self._init_ollama(model_identifier)
        else:
            self.backend = "hf"
            self._init_huggingface(model_identifier)
        
        print(f"Model '{model_identifier}' on backend '{self.backend}' initialized in {time.time() - t_start:.2f}s")

    def _init_ollama(self, model_identifier: str):
        """Initializes the Ollama backend."""
        if ollama is None:
            raise ImportError("The 'ollama' package is not installed. Please install it with `pip install ollama`.")
        
        self.ollama_model_name = model_identifier.split("/", 1)[1]
        try:
            # Check if the Ollama server is running and the model exists.
            self.client = ollama.Client()
            self.client.show(self.ollama_model_name)
            print(f"Successfully connected to Ollama and verified model '{self.ollama_model_name}'.")
        except Exception as e:
            print(f"Error: Could not connect to Ollama or find model '{self.ollama_model_name}'.")
            print(f"Please ensure the Ollama server is running and you have pulled the model (e.g., `ollama pull {self.ollama_model_name}`).")
            raise e

    def _init_huggingface(self, model_identifier: str):
        """Initializes the legacy Hugging Face backend."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Warning: Using legacy Hugging Face backend. This path is for older span-infilling models like InCoder.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_identifier).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Legacy placeholders for InCoder
        self.infill_ph = "<|mask:0|>"
        self.extra_end = "<|mask:1|><|mask:0|>"
        self.EOM = "<|endofmask|>"

    def generate(self, infill_code: str, num_samples: int, do_sample: bool = True) -> list[str]:
        """
        Generates code completions for the given input.
        
        Args:
            infill_code: The code with a placeholder (e.g., <FILL_ME>).
            num_samples: The number of code samples to generate.
            do_sample: Whether to use sampling (recommended for diversity).

        Returns:
            A list of completed code strings.
        """
        if self.infill_ph not in infill_code:
            print(f"Warning: Placeholder '{self.infill_ph}' not found in the input code. Returning no generations.")
            return []

        if self.backend == "ollama":
            return self._generate_ollama(infill_code, num_samples, do_sample)
        else: # self.backend == "hf"
            return self._generate_huggingface(infill_code, num_samples, do_sample)

    def _generate_ollama(self, infill_code: str, num_samples: int, do_sample: bool) -> list[str]:
        """Generates code using the Ollama backend."""
        prefix, suffix = infill_code.split(self.infill_ph, 1)

        # A clear, robust prompt for modern instruction-tuned models
        prompt_template = f"""You are an expert Python programmer. Your task is to complete the following Python code snippet. The missing part is indicated by `{self.infill_ph}`.
        You must provide ONLY the raw Python code that replaces `{self.infill_ph}`. Do not add any explanations, comments, or markdown formatting like ```python ... ```.

        Code snippet:
        ```python
        {infill_code}
        ```
        Provide the code to fill in {self.infill_ph}:
        """
        outputs = []
        for _ in range(num_samples):
            try:
                # Temperature and top_p are common sampling parameters
                options = {"temperature": 1.0, "top_p": 0.95} if do_sample else {}
                response = self.client.generate(
                    model=self.ollama_model_name,
                    prompt=prompt_template,
                    stream=False,
                    options=options,
                )
                completion = response['response']
                
                # Clean the output: remove markdown and surrounding whitespace
                completion = re.sub(r'```(?:python\n)?(.*?)\n?```', r'\1', completion, flags=re.DOTALL).strip()
                
                outputs.append(prefix + completion + suffix)
            except Exception as e:
                print(f"Error during Ollama generation: {e}")
                continue
        return outputs

    def _generate_huggingface(self, infill_code: str, num_samples: int, do_sample: bool) -> list[str]:
        """Generates code using the legacy Hugging Face backend."""
        import torch
        
        prompt = infill_code + self.extra_end
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        max_length = inputs.input_ids.shape[1] + 64 # Generate up to 64 new tokens
        
        with torch.no_grad():
            raw_outputs = self.model.generate(
                **inputs,
                do_sample=do_sample,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=num_samples,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        decoded_outputs = self.tokenizer.batch_decode(raw_outputs, skip_special_tokens=False)
        
        completions = []
        for output in decoded_outputs:
            # Extract the generated text between the prompt and the end-of-mask token
            start_index = len(prompt)
            try:
                end_index = output.index(self.EOM, start_index)
                completion = output[start_index:end_index]
                completions.append(infill_code.replace(self.infill_ph, completion))
            except ValueError:
                continue # EOM token not found, generation was likely incomplete
        
        return completions

