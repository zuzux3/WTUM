import gradio as gr

def exampleFunc():
    return "Picture"

demo = gr.Interface(
    fn = exampleFunc
    inputs='image'
    outputs='label'
)

demo.launch(share=True)