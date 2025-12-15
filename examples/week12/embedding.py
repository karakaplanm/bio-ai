"""
Image Embedding Generator using ResNet34 ONNX Model
Generates 512-dimensional vector embeddings for images.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import os
from typing import List, Union


class ImageEmbedder:
    """Generate 512-dimensional embeddings using ResNet34 ONNX model."""
    
    def __init__(self, model_path: str = "models/resnet34-v2-7.onnx"):
        """
        Initialize the embedder with ResNet34 ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.input_size = (224, 224)
        self.embedding_dim = 512
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Load the ONNX model
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and configure for 512-dim embedding extraction."""
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get input name
        self.input_name = self.session.get_inputs()[0].name
        
        # Get output dimension (1000 for ImageNet)
        output_shape = self.session.get_outputs()[0].shape
        self.output_dim = output_shape[-1] if output_shape[-1] else 1000
        
        # Create projection matrix to reduce 1000 -> 512 dimensions
        self.projection_matrix = self._create_projection_matrix(self.output_dim, self.embedding_dim)
    
    def _create_projection_matrix(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Create a random orthogonal projection matrix for dimensionality reduction."""
        np.random.seed(42)  # For reproducibility
        random_matrix = np.random.randn(input_dim, output_dim).astype(np.float32)
        # Orthogonalize using QR decomposition
        q, _ = np.linalg.qr(random_matrix)
        return q[:, :output_dim]
    
    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for the model.
        
        Args:
            image: Path to image, PIL Image, or numpy array
            
        Returns:
            Preprocessed image tensor of shape (1, 3, 224, 224)
        """
        # Load image if path provided
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        # Resize to input size
        img = img.resize(self.input_size, Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Apply ImageNet normalization
        img_array = (img_array - self.mean) / self.std
        
        # Transpose to (C, H, W) and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def get_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Generate 512-dimensional embedding for an image.
        
        Args:
            image: Path to image, PIL Image, or numpy array
            
        Returns:
            512-dimensional embedding vector
        """
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Get 1000-dim output from ResNet34
        outputs = self.session.run(
            None,
            {self.input_name: input_tensor}
        )
        features = outputs[0].flatten()
        
        # Project to 512 dimensions
        embedding = np.dot(features, self.projection_matrix)
        
        # L2 normalize the embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def get_embeddings_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of image paths, PIL Images, or numpy arrays
            
        Returns:
            Array of shape (N, 512) containing embeddings
        """
        embeddings = []
        for image in images:
            embedding = self.get_embedding(image)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        return float(np.dot(embedding1, embedding2))


def main():
    """Example usage of the ImageEmbedder."""
    # Initialize embedder
    embedder = ImageEmbedder("models/resnet34-v2-7.onnx")
    
    # Get all images from the images directory
    images_dir = "images"
    image_files = [
        os.path.join(images_dir, f) 
        for f in os.listdir(images_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    print(f"Found {len(image_files)} images")
    print("-" * 50)
    
    # Generate embeddings for each image
    embeddings = {}
    for image_path in image_files:
        embedding = embedder.get_embedding(image_path)
        embeddings[image_path] = embedding
        print(f"Image: {os.path.basename(image_path)}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding (first 10 values): {embedding[:10]}")
        print()
    
    # Compute pairwise similarities
    if len(image_files) > 1:
        print("Pairwise Similarities:")
        print("-" * 50)
        for i, path1 in enumerate(image_files):
            for path2 in image_files[i+1:]:
                similarity = embedder.compute_similarity(
                    embeddings[path1], 
                    embeddings[path2]
                )
                print(f"{os.path.basename(path1)} <-> {os.path.basename(path2)}: {similarity:.4f}")


if __name__ == "__main__":
    main()
