import cv2
import torch
import matplotlib.pyplot as plt


midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cuda")
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform
plt.axis("off")
plt.title("Hold q to Exit")
camera = cv2.VideoCapture(0)
while camera.isOpened(): 
    ret, frame = camera.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to("cuda")
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),
            size = img.shape[:2], mode="bicubic", align_corners=False).squeeze()
        output = prediction.cpu().numpy()
    cv2.imshow("Hold q to Exit", frame)
    plt.imshow(output)
    plt.pause(0.000001)
    key = cv2.waitKey(33) & 0b11111111
    if key == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
