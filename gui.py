import gradio as gr
from classification import classify

input = gr.components.Image(label="Load Your Picture")
outputRes18 = gr.components.Label(label='Resnet18 Prediction')
outputRes50 = gr.components.Label(label='Resnet50 Prediction')

output = [outputRes18, outputRes50]


demo = gr.Interface(
    fn = classify,
    inputs=input,
    outputs=output)

demo.launch()