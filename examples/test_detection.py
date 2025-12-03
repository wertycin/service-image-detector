import base64
import httpx

IMAGE_URL = "https://kchr.topnomer.ru/media/uploads/images/2023/07/QW7-0Khqv_I.webp"
SERVICE_URL = "http://localhost:8000/forward_with_show"


def main():
    # Скачиваем и сохраняем оригинальное изображение
    print(f"Downloading original image from {IMAGE_URL}...")
    response = httpx.get(IMAGE_URL)
    response.raise_for_status()
    with open("examples/data/input.png", "wb") as f:
        f.write(response.content)
    print("Saved original image to examples/data/input.png")

    # Отправляем запрос в сервис детекции
    print(f"Sending request to {SERVICE_URL}...")
    response = httpx.post(
        SERVICE_URL,
        json={"url": IMAGE_URL},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    # Выводим обнаруженные bboxes
    print(f"\nDetected {len(result['bboxes'])} objects:")
    for bbox in result["bboxes"]:
        print(f"  - {bbox['label']}: {bbox['confidence']:.2f} "
              f"({bbox['top_left']['x']:.0f}, {bbox['top_left']['y']:.0f}) -> "
              f"({bbox['bottom_right']['x']:.0f}, {bbox['bottom_right']['y']:.0f})")

    # Декодируем и сохраняем итоговое изображение
    image_base64 = result["image"]
    image_bytes = base64.b64decode(image_base64)
    with open("examples/data/output.png", "wb") as f:
        f.write(image_bytes)
    print("\nSaved annotated image to examples/data/output.png")


if __name__ == "__main__":
    main()


