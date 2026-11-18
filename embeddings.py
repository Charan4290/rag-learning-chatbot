
from sentence_transformers import SentenceTransformer, util

# Load our pretrained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Three employee questions
sentence1 = "How do I reset my password?"
sentence2 = "I forgot my login credentials."
sentence3 = "How many vacation days do I have?"

# Convert sentences into embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
embedding3 = model.encode(sentence3, convert_to_tensor=True)

# Calculate similarity scores
score1 = util.cos_sim(embedding1, embedding2).item()
score2 = util.cos_sim(embedding1, embedding3).item()

print("Password vs Login:", round(score1, 3))
print("Password vs Vacation:", round(score2, 3))
