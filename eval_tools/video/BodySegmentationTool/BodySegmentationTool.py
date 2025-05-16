import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
import sys

sys.path.append('./face-parsing.PyTorch')
from model import BiSeNet

class FaceAnalyzer:
    def __init__(self, device="cuda"):
        # Set computation device (CPU or GPU)
        self.device = torch.device(device)
        self._init_models()
        
    def _init_models(self):
        # Initialize the body segmentation model
        # Model: DeepLabV3 with ResNet-101 backbone, pretrained on COCO dataset for person segmentation
        self.body_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.body_model = self.body_model.eval().to(self.device)
        
        # Initialize the face parsing model
        # Model: BiSeNet (Bi-Directional Spatial Context Aggregation Network)
        # Number of classes: 19 facial regions
        model_path = r"79999_iter.pth"  # Path to pretrained BiSeNet weights
        self.face_model = BiSeNet(n_classes=19)
        self.face_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.face_model = self.face_model.eval().to(self.device)
        
        # Set up preprocessing for face parsing
        # Transform: Resize to 512x512, normalize with ImageNet mean/std
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),  # mean values for R, G, B channels
                (0.229, 0.224, 0.225)   # std dev values for R, G, B channels
            )
        ])
        
        # Set up preprocessing for body segmentation
        # Transform: Resize to 520x520 for DeepLabV3 input, normalize with ImageNet mean/std
        self.body_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),  # DeepLabV3 expected input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # mean values for R, G, B channels
                std=[0.229, 0.224, 0.225]    # std dev values for R, G, B channels
            )
        ])

    def _get_face_bbox(self, mask):
        """Extract bounding rectangles for faces from the segmentation mask."""
        # Morphological closing (kernel size 5×5) to refine the binary mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and filter by area >100 pixels
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) > 100  # minimum area threshold
        ]

    def analyze(self, image_path):
        # Load the image from disk
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Body segmentation
        body_tensor = self.body_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            body_output = self.body_model(body_tensor)['out'][0]
            # Class index 15 corresponds to “person” in COCO
            body_mask = (body_output.argmax(0) == 15).cpu().numpy()
        
        body_mask = cv2.resize(
            body_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Face parsing
        face_img = cv2.resize(img, (512, 512))  # BiSeNet input size
        face_tensor = self.face_transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            face_out = self.face_model(face_tensor)[0]
        
        face_mask = face_out.argmax(1).squeeze().cpu().numpy()
        face_mask = cv2.resize(
            face_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        # Keep only relevant facial region classes
        face_mask = np.isin(face_mask, [1,2,3,4,5,6,10,12,13]).astype(np.uint8)
        
        # Extract face bounding boxes
        face_boxes = self._get_face_bbox(face_mask)
        
        # Compute area percentages
        total_pixels = h * w
        body_percent = body_mask.sum() / total_pixels * 100
        face_percent = face_mask.sum() / total_pixels * 100
        
        # Print summary of results
        print("\nAnalysis results:")
        print(f"Body area: {body_percent:.2f}%")
        print(f"Face area: {face_percent:.2f}%")
        print(f"Number of faces detected: {len(face_boxes)}")
        
        return {
            'image': img,
            'body_mask': body_mask,
            'face_mask': face_mask,
            'face_boxes': face_boxes,
            'body_percent': body_percent,
            'face_percent': face_percent
        }

def visualize(results):
    img = results['image']
    h, w = img.shape[:2]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # Overlay body mask in red with 40% opacity
    rgba = np.zeros((h, w, 4))
    rgba[..., 0] = 1.0  # red channel
    rgba[..., 3] = results['body_mask'] * 0.4  # alpha mask
    ax.imshow(rgba)
    
    # Draw body contours (yellow dashed)
    body_contours, _ = cv2.findContours(
        results['body_mask'].astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in body_contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 2:
            ax.plot(cnt[:,0], cnt[:,1], '--', lw=2, color='yellow', label='Body')
    
    # Draw face bounding boxes (dodgerblue)
    for (x, y, fw, fh) in results['face_boxes']:
        rect = plt.Rectangle((x, y), fw, fh,
                             ec='dodgerblue', fill=False, lw=2,
                             label='Face Box')
        ax.add_patch(rect)
        ax.text(x, y-10, f"Face ({fw}×{fh})",
                color='dodgerblue', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5))
    
    # Draw face region contours (lime dotted)
    face_contours, _ = cv2.findContours(
        results['face_mask'],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in face_contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 2:
            ax.plot(cnt[:,0], cnt[:,1], ':', lw=2, color='lime', label='Face')
    
    # Add title with statistics
    stats = [
        f"Body Area: {results['body_percent']:.1f}%",
        f"Faces: {len(results['face_boxes'])}",
        f"Face Area: {results['face_percent']:.1f}%"
    ]
    ax.set_title("\n".join(stats), fontsize=14, pad=20)
    ax.axis('off')
    
    # Consolidate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               loc='upper right', fontsize=10,
               facecolor='black', edgecolor='white')
    
    plt.tight_layout()
    plt.savefig('result.jpg', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Choose CUDA GPU if available, else CPU
    analyzer = FaceAnalyzer(device="cuda" if torch.cuda.is_available() else "cpu")
    results = analyzer.analyze("input.jpg")
    visualize(results)
