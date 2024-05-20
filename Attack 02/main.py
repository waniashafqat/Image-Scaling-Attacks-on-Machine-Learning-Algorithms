import ScalingAttack as sa
from PIL import Image

if __name__ == "__main__":
    source_image = None
    target_image = None
    try:
        # Open the source image
        source_image = Image.open("sheep.jpg")
        source_image_name = "sheep.jpg"

        # Open the target image
        target_image = Image.open("wolf.jpg")
        target_image_name = "wolf.jpg"

    except IOError:
        # Handle file not found error
        print("File Not Found")
        exit(-1)

    # Print source and target image details
    print(f"\nSource Image: {source_image_name}")
    print(f"Dimensions: {source_image.size} (width, height)")

    print(f"\nTarget Image: {target_image_name}")
    print(f"Dimensions: {target_image.size} (width, height)")

    # Print message indicating attack generation
    print("\nGenerating Image Scaling Attack ....")

    # Implement the attack
    sa.implement_attack(source_image, target_image)

    # Print success message
    print("\nAttack successfully completed.")
    print(f"Attack image saved as 'attack_image.jpg'")
