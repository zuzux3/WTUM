import gradio as gr
from classification import classify

input = gr.components.Image(label="Load Your Picture")
outputCNN = gr.components.Label(label='CNN Prediction')
outputVGG16 = gr.components.Label(label='VGG16 Prediction')
outputRes50 = gr.components.Label(label='Resnet50 Prediction')

output = [outputVGG16, outputRes50]


demo = gr.Interface(
    fn = classify,
    inputs=input,
    outputs=output)

demo.launch()