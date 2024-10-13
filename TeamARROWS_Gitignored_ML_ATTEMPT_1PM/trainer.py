import os
import torch
import requests
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
import re

# Set expandable memory segments for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the model and processor once
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,  # Mixed precision for efficiency
    device_map="auto"  # Automatically place on GPU if available
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


def download_image_from_url(image_url):
    """Download an image from a URL and return it as a PIL Image."""
    response = requests.get(image_url)
    response.raise_for_status()  # Check for request errors
    image = Image.open(BytesIO(response.content))
    return image


def process_image(image):
    """Resize and process the image for model input."""
    image = image.convert('RGB')  # Ensure correct color mode
    image = image.resize((512, 512))  # Resize to save memory
    return image


def clean_response(response):
    """Extract the precise numerical value and its unit from the response, and convert units to full form."""
    # Define a dictionary to map short forms to full forms
    unit_conversion = {
        'cm': 'centimeter', 'mm': 'millimeter', 'm': 'meter', 'g': 'gram',
        'kg': 'kilogram', 'oz': 'ounce', 'lb': 'pound', 'kv': 'kilovolt',
        'mv': 'millivolt', 'v': 'volt', 'kw': 'kilowatt', 'w': 'watt',
        'ml': 'milliliter', 'l': 'liter', 'ft': 'foot', 'yd': 'yard', 'in': 'inch'
    }

    # Build a regex pattern to match number followed by a short unit
    short_units_pattern = "|".join(unit_conversion.keys())
    pattern = rf"(\d+(\.\d+)?)\s*({short_units_pattern})"

    # Search for a match in the response
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        # Extract the value and the short form unit
        value = match.group(1)  # The numeric value (e.g., '15')
        short_unit = match.group(3).lower()  # The short unit form (e.g., 'in')

        # Convert to full form using the dictionary
        full_form_unit = unit_conversion.get(short_unit, short_unit)

        # Return the value followed by the full form of the unit (no abbreviation)
        return f"{value} {full_form_unit}"
    else:
        return "Value not found"  # If no match is found, return a default message


def generate_response(image, text_query):
    """Generate a response for the given image and text query."""
    # Define the message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_query}
            ]
        }
    ]

    # Apply chat template for the text prompt
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # Generate output
    with torch.no_grad():  # Disable gradient computation for inference
        output_ids = model.generate(**inputs, max_new_tokens=256)  # Limit token generation

    # Decode the generated output text
    output_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Clean the response to extract the precise value
    clean_output = clean_response(output_text[0].strip())

    return clean_output


def process_csv_and_generate_responses(csv_file, output_csv_file):
    """Process a CSV file, generate responses for each image and entity, and save results."""
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create a new column for the responses
    df['response'] = ""

    # Iterate through each row
    for index, row in df.iterrows():
        image_url = row['image_link']
        entity_name = row['entity_name']

        try:
            # Download and process image from URL
            image = download_image_from_url(image_url)
            processed_image = process_image(image)

            # Generate response with the query based on entity_name
            query = f"What is {entity_name}?"
            response = generate_response(processed_image, query)

            # Add the response to the DataFrame
            df.at[index, 'response'] = response

            # Clear GPU cache and remove variables to free memory
            torch.cuda.empty_cache()
            del image, processed_image
        except Exception as e:
            # If there's an error with the image or query, log it in the response
            df.at[index, 'response'] = f"Error: {str(e)}"
            torch.cuda.empty_cache()

    # Save the DataFrame with the new 'response' column to a new CSV file
    df.to_csv(output_csv_file, index=False)
    print(f"Results saved to {output_csv_file}")


def process_multiple_csv_files():
    """Process multiple CSV files, one at a time, and save results after each."""
    for i in range(11, 21):
        input_csv = f"C:\\Users\\prath\\PycharmProjects\\ML_ATTEMPT_FINAL\\csvSplit\\test{i}.csv"
        output_csv = f"C:\\Users\\prath\\PycharmProjects\\ML_ATTEMPT_FINAL\\Trained\\test{i}_output.csv"

        print(f"Processing CSV file {i}/10: {input_csv}")
        process_csv_and_generate_responses(input_csv, output_csv)
        print(f"CSV file {i} processed and saved as {output_csv}")

        # Clear any remaining GPU cache or variables to free memory
        torch.cuda.empty_cache()


# Process all 10 CSV files
process_multiple_csv_files()

# Clear GPU cache to release
torch.cuda.empty_cache()