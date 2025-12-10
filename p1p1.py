import requests
from PIL import Image
from io import BytesIO
from config import HF_API_KEY
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

def generate_image(prompt):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        return image

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
def main ():
    prompt = input("Enter a prompt for image generation: ")
    image = generate_image(prompt)
    
    if image:
        image.show()
        save_option = input("Do you want to save the image? (y/n): ").strip().lower()
        if save_option == 'y':
            image.save("generated_image.png")
            print("Image saved as 'generated_image.png'")

if __name__ == "__main__":
    main()