# Model contract

Sensor embeddings are 64-D, text embeddings are 768-D, and the fusion head requires explicit modality presence. A forward pass does not establish training. The API returns unavailable status when no verified fusion checkpoint exists.
