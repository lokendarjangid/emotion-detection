from deepface import DeepFace

face = DeepFace.analyze('img1.jpg',actions=['emotion'])
print(face[0]['dominant_emotion'])