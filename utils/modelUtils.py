import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import gc
import numpy as np

class ModelUtils:
    """
    A utility class for loading and utilizing a transformer-based model for sequence classification.

    Attributes:
        model_name (str): The name of the model to be loaded.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the specified model.
        config (AutoConfig): Configuration for the model, with output_hidden_states enabled.
        model (torch.nn.Module): The loaded transformer model, either on CPU or GPU.

    Methods:
        load_model(): Loads the model into an available device (GPU or CPU).
        model_predict(texts): Predicts the class probabilities for a list of text inputs.
        get_embeddings(texts): Extracts embeddings for a list of text inputs using the model's last hidden state.
    """

    def __init__(self, model_name: str):
        """
        Initializes the ModelUtils class by setting the model name, and loading the tokenizer and config.

        Parameters:
            model_name (str): The name of the pre-trained model to use.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        """
        Loads the model specified by the model name into the most suitable device (GPU or CPU).

        Returns:
            torch.nn.Module: The loaded model ready for inference.
        """
        gc.collect()
        torch.cuda.empty_cache()
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config).to('cuda')
            print("Classification model loaded on CUDA")
        except RuntimeError:
            print("CUDA out of memory. Loading classification model on CPU instead.")
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config).to('cpu')
        model.eval()
        return model

    def model_predict(self, texts: list) -> np.ndarray:
        """
        Predicts class probabilities for given texts using the model.

        Parameters:
            texts (list of str): A list of text strings to predict.

        Returns:
            np.ndarray: An array of class probabilities for each input text.
        """
        predictions = []
        device = self.model.device
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                #probs = probs[:, [0, 2, 1]]  # Reorder to ["True", "False", "Not Enough Info"]
            predictions.append(probs)
        return np.array(predictions)

    def get_embeddings(self, texts: list) -> np.ndarray:
        """
        Extracts embeddings from the last hidden state of the model for a list of texts.

        Parameters:
            texts (list of str): A list of text strings from which to extract embeddings.

        Returns:
            np.ndarray: An array of embeddings for each input text.
        """
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.model.device)
                outputs = self.model(**inputs)
                emb = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
                embeddings.append(emb.flatten())
        return np.array(embeddings)