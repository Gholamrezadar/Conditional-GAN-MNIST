import torch
import gradio as gr
from models import Generator
from conditional_gan import generate_digit

generator = Generator()

def init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the generator
    generator.load_state_dict(torch.load('generator.pt', map_location=device))
    generator.to(device)

def generate_mnist_digit(digit):
    return generate_digit(generator, digit)

# Gradio Interface
def gradio_generate(digit):
    return generate_mnist_digit(digit)

with gr.Blocks() as demo:
    gr.Markdown("# MNIST Digit Generator")
    digit = gr.Dropdown(list(range(10)), label="Select a Digit")
    generate_button = gr.Button("Generate")
    output_image = gr.Image(label="Generated Image", type="filepath")
    
    generate_button.click(gradio_generate, inputs=digit, outputs=output_image)

if __name__ == '__main__':
    init()
    print("* Model loaded")
    demo.launch()
