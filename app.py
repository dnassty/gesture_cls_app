from fastai.vision.all import *
import gradio as gr

learn = load_learner('thumbs_vs_heart_model.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


title = "Hand Gesture Classifier"
description = "A hand gesture classifier trained for two classes thumbs up and heart gesture. Created as a demo for Gradio and HuggingFace Spaces."

examples = ['boy_thumbs_up.jpg', 'girl_thumbs_up.jpg', 'man_heart.jpg', 'heart.jpg']
gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=2), title=title, description=description, examples=examples,
             interpretation='default').launch(share=True)