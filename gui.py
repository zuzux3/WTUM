import gradio as gr

def exampleFunc(image):
    return "Picture"

demo = gr.Interface(
    fn = exampleFunc,
    inputs='image',
    outputs='label'
)

demo.launch()