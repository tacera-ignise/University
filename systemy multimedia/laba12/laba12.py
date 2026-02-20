
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import json


def saveImage(image: np.ndarray, filename: str):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


class VectorRasterizer:
    def __init__(self, width: int, height: int, layers: int = 3):
        self.width = width
        self.height = height
        self.layers = layers
        self.canvas = np.zeros((height, width, layers), dtype=np.uint8)
        self.buffer = np.zeros_like(self.canvas, dtype=int) - 99

    def clearCanvas(self):
        self.canvas = np.zeros((self.height, self.width, self.layers), dtype=np.uint8)
        self.buffer = np.zeros_like(self.canvas, dtype=int) - 99

    def setPixel(self, x: int, y: int, color: List[int]):
        if 0 <= x < self.width and 0 <= y < self.height:
            for i, c in enumerate(color[: self.layers]):
                self.canvas[y, x, i] = c

    def drawLineBresenham(
        self, p1: Tuple[int, int], p2: Tuple[int, int], color: List[int]
    ):
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        xInc = 1 if x1 < x2 else -1
        yInc = 1 if y1 < y2 else -1

        if dx >= dy:
            p = 2 * dy - dx
            for _ in range(dx + 1):
                self.setPixel(x, y, color)
                if p < 0:
                    p += 2 * dy
                else:
                    p += 2 * dy - 2 * dx
                    y += yInc
                x += xInc
        else:
            p = 2 * dx - dy
            for _ in range(dy + 1):
                self.setPixel(x, y, color)
                if p < 0:
                    p += 2 * dx
                else:
                    p += 2 * dx - 2 * dy
                    x += xInc
                y += yInc

    def drawCircleBresenham(
        self, center: Tuple[int, int], radius: int, color: List[int]
    ):
        cx, cy = center
        x = 0
        y = radius
        p = 5 / 4 - radius

        while x <= y:
            points = [
                (cx + x, cy + y),
                (cx - x, cy + y),
                (cx + x, cy - y),
                (cx - x, cy - y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx + y, cy - x),
                (cx - y, cy - x),
            ]

            for px, py in points:
                self.setPixel(px, py, color)

            if p < 0:
                p += 2 * x + 1
            else:
                p += 2 * x + 1 - 2 * y
                y -= 1
            x += 1

    def drawTriangle(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        p3: Tuple[int, int],
        color: List[int],
    ):
        self.drawLineBresenham(p1, p2, color)
        self.drawLineBresenham(p2, p3, color)
        self.drawLineBresenham(p3, p1, color)

    def drawRectangle(self, p1: Tuple[int, int], p2: Tuple[int, int], color: List[int]):
        x1, y1 = p1
        x2, y2 = p2
        self.drawLineBresenham((x1, y1), (x2, y1), color)
        self.drawLineBresenham((x2, y1), (x2, y2), color)
        self.drawLineBresenham((x2, y2), (x1, y2), color)
        self.drawLineBresenham((x1, y2), (x1, y1), color)

    def drawPolygon(self, points: List[Tuple[int, int]], color: List[int]):
        if len(points) < 3:
            return

        for i in range(len(points)):
            nextI = (i + 1) % len(points)
            self.drawLineBresenham(points[i], points[nextI], color)

    def fillPolygon(self, points: List[Tuple[int, int]], color: List[int]):
        if len(points) < 3:
            return

        minY = min(p[1] for p in points)
        maxY = max(p[1] for p in points)

        for y in range(max(0, minY), min(self.height, maxY + 1)):
            intersections = []

            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]

                if p1[1] != p2[1]:
                    if min(p1[1], p2[1]) <= y < max(p1[1], p2[1]):
                        x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                        intersections.append(int(x))

            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x1 = max(0, intersections[i])
                    x2 = min(self.width - 1, intersections[i + 1])
                    for x in range(x1, x2 + 1):
                        self.setPixel(x, y, color)

    def drawFilledCircle(self, center: Tuple[int, int], radius: int, color: List[int]):
        cx, cy = center
        for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    self.setPixel(x, y, color)

    def drawStar(
        self,
        center: Tuple[int, int],
        outerRadius: int,
        innerRadius: int,
        color: List[int],
        points: int = 5,
    ):
        cx, cy = center
        starPoints = []

        import math

        for i in range(points * 2):
            angle = i * math.pi / points
            radius = outerRadius if i % 2 == 0 else innerRadius
            x = int(cx + radius * math.cos(angle - math.pi / 2))
            y = int(cy + radius * math.sin(angle - math.pi / 2))
            starPoints.append((x, y))

        self.fillPolygon(starPoints, color)

    def drawDiamond(self, center: Tuple[int, int], size: int, color: List[int]):
        cx, cy = center
        points = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
        self.fillPolygon(points, color)


def generateImageFromDict(imageData: Dict[str, Any]) -> np.ndarray:
    canvasInfo = imageData["Canvas"]
    width = canvasInfo["width"]
    height = canvasInfo["height"]
    bgColor = canvasInfo.get("background_color", [0, 0, 0])

    rasterizer = VectorRasterizer(width, height)

    if bgColor != [0, 0, 0]:
        rasterizer.canvas[:, :, :] = bgColor

    shapes = sorted(imageData["Shapes"], key=lambda x: x.get("Z_layer", 0))

    for shape in shapes:
        shapeType = shape["type"]
        color = shape["color"]

        if shapeType == "line":
            rasterizer.drawLineBresenham(tuple(shape["p1"]), tuple(shape["p2"]), color)

        elif shapeType == "circle":
            if shape.get("filled", False):
                rasterizer.drawFilledCircle(
                    tuple(shape["center"]), shape["radius"], color
                )
            else:
                rasterizer.drawCircleBresenham(
                    tuple(shape["center"]), shape["radius"], color
                )

        elif shapeType == "filled_circle":
            rasterizer.drawFilledCircle(tuple(shape["center"]), shape["radius"], color)

        elif shapeType == "rectangle":
            if shape.get("filled", False):
                x1, y1 = shape["p1"]
                x2, y2 = shape["p2"]
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                rasterizer.fillPolygon(points, color)
            else:
                rasterizer.drawRectangle(tuple(shape["p1"]), tuple(shape["p2"]), color)

        elif shapeType == "filled_rectangle":
            x1, y1 = shape["p1"]
            x2, y2 = shape["p2"]
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            rasterizer.fillPolygon(points, color)

        elif shapeType == "triangle":
            if shape.get("filled", False):
                points = [tuple(shape["p1"]), tuple(shape["p2"]), tuple(shape["p3"])]
                rasterizer.fillPolygon(points, color)
            else:
                rasterizer.drawTriangle(
                    tuple(shape["p1"]), tuple(shape["p2"]), tuple(shape["p3"]), color
                )

        elif shapeType == "filled_triangle":
            points = [tuple(shape["p1"]), tuple(shape["p2"]), tuple(shape["p3"])]
            rasterizer.fillPolygon(points, color)

        elif shapeType == "polygon":
            if shape.get("filled", False):
                rasterizer.fillPolygon([tuple(p) for p in shape["points"]], color)
            else:
                rasterizer.drawPolygon([tuple(p) for p in shape["points"]], color)

        elif shapeType == "filled_polygon":
            rasterizer.fillPolygon([tuple(p) for p in shape["points"]], color)

        elif shapeType == "star":
            rasterizer.drawStar(
                tuple(shape["center"]),
                shape["outer_radius"],
                shape["inner_radius"],
                color,
                shape.get("points", 5),
            )

        elif shapeType == "diamond":
            rasterizer.drawDiamond(tuple(shape["center"]), shape["size"], color)

    return rasterizer.canvas


def createComplexImageStructure(
    width: int, height: int, studentLastDigit: int = 0
) -> Dict[str, Any]:
    imageData = {
        "Canvas": {"width": width, "height": height, "background_color": [20, 20, 30]},
        "Shapes": [],
    }

    imageData["Shapes"].extend(
        [
            {
                "type": "filled_circle",
                "Z_layer": 1,
                "center": [width // 4, height // 4],
                "radius": min(width, height) // 8,
                "color": [100, 150, 200],
            },
            {
                "type": "filled_triangle",
                "Z_layer": 2,
                "p1": [width // 4 - 30, height // 4 + 20],
                "p2": [width // 4 + 50, height // 4 + 20],
                "p3": [width // 4 + 10, height // 4 - 40],
                "color": [200, 100, 100],
            },
        ]
    )

    rectX, rectY = width // 2, height // 4
    rectW, rectH = width // 6, height // 8

    imageData["Shapes"].extend(
        [
            {
                "type": "filled_rectangle",
                "Z_layer": 1,
                "p1": [rectX - rectW // 2, rectY - rectH // 2],
                "p2": [rectX + rectW // 2, rectY + rectH // 2],
                "color": [80, 120, 80],
            },
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [rectX - rectW // 4, rectY - rectH // 4],
                "p2": [rectX, rectY + rectH // 4],
                "color": [150, 150, 50],
            },
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [rectX, rectY - rectH // 4],
                "p2": [rectX + rectW // 4, rectY + rectH // 4],
                "color": [150, 50, 150],
            },
        ]
    )

    lX, lY = 3 * width // 4, height // 4
    lSize = min(width, height) // 12

    lPoints = [
        [lX, lY],
        [lX + lSize, lY],
        [lX + lSize, lY + lSize // 2],
        [lX + lSize // 2, lY + lSize // 2],
        [lX + lSize // 2, lY + lSize],
        [lX, lY + lSize],
    ]

    imageData["Shapes"].append(
        {
            "type": "filled_polygon",
            "Z_layer": 1,
            "points": lPoints,
            "color": [200, 150, 100],
        }
    )

    brownRectX, brownRectY = width // 3, 2 * height // 3
    centerX, centerY = width // 3, height // 2
    margin = 20
    size = min(width, height) // 6
    baseX, baseY = margin + size // 2, height - margin - size // 2
    imageData["Shapes"].extend(
        [
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [brownRectX - 40, brownRectY - 30],
                "p2": [brownRectX + 40, brownRectY + 30],
                "color": [101, 67, 33],
            },
            {
                "type": "filled_circle",
                "Z_layer": 1,
                "center": [brownRectX - 20, brownRectY],
                "radius": 25,
                "color": [255, 255, 0],
            },
        ]
    )
    offsetX = 2 * size + 5 * margin
    digitShapes = {
        0: [
            {
                "type": "filled_circle",
                "Z_layer": 1,
                "center": [2 * width // 3, 2 * height // 3],
                "radius": 30,
                "color": [0, 0, 255],
            },
            {
                "type": "filled_circle",
                "Z_layer": 2,
                "center": [2 * width // 3 + 25, 2 * height // 3],
                "radius": 30,
                "color": [255, 255, 255],
            },
            {
                "type": "filled_circle",
                "Z_layer": 3,
                "center": [2 * width // 3 + 50, 2 * height // 3],
                "radius": 30,
                "color": [255, 255, 0],
            },
        ],
     
        8: 
        [
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [centerX - size // 2 + offsetX, centerY - size // 2  +70],
                "p2": [centerX + size // 2 + offsetX, centerY + size  // 2  +70],
                "color": [255, 255, 0], 
            },
            {
                "type": "filled_polygon",
                "Z_layer": 3,
                "points": [
                    [centerX + offsetX, centerY - size  // 2 +20],     
                    [centerX + size // 2 + offsetX, centerY +20],    
                    [centerX + offsetX, centerY + size  // 2 +20],   
                    [centerX - size // 2 + offsetX, centerY +20],   
                ],
                "color": [0, 0, 0],
            },
            {
                "type": "filled_circle",
                "Z_layer": 1,
                "center": [centerX - size - margin + offsetX, centerY +40],
                "radius": size // 2,
                "color": [255, 0, 0],  
            },
        ]
        
    }

    if studentLastDigit in digitShapes:
        imageData["Shapes"].extend(digitShapes[studentLastDigit])

    return imageData


def calculateMse(img1: np.ndarray, img2: np.ndarray) -> float:
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def calculateSsim(img1: np.ndarray, img2: np.ndarray) -> float:
    if len(img1.shape) == 3:
        img1Gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2Gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1Gray = img1
        img2Gray = img2

    return ssim(img1Gray, img2Gray)


def main():
    testRasterizer = VectorRasterizer(200, 200)
    testRasterizer.drawLineBresenham((10, 10), (100, 50), [255, 0, 0])
    testRasterizer.drawCircleBresenham((100, 100), 30, [0, 255, 0])
    testRasterizer.drawFilledCircle((150, 150), 20, [0, 0, 255])
    testRasterizer.drawTriangle((50, 150), (100, 120), (80, 180), [255, 255, 0])
    testRasterizer.drawRectangle((120, 20), (180, 80), [255, 0, 255])


    testImageData = {
        "Canvas": {"width": 400, "height": 300, "background_color": [50, 50, 50]},
        "Shapes": [
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [100, 100],
                "p2": [200, 200],
                "color": [0, 0, 255],  
            },
            {
                "type": "filled_circle",
                "Z_layer": 1,  
                "center": [120, 180],
                "radius": 40,
                "color": [255, 165, 0],  
            },
            {
                "type": "filled_circle",
                "Z_layer": 3,  
                "center": [180, 120],
                "radius": 40,
                "color": [255, 255, 255],  
            },
            {
                "type": "filled_circle",
                "Z_layer": 1,
                "center": [300, 100],
                "radius": 60,
                "color": [0, 255, 0],  
            },
            {
                "type": "filled_triangle",
                "Z_layer": 2,
                "p1": [270, 70],
                "p2": [330, 70],
                "p3": [300, 130],
                "color": [255, 0, 0], 
            },
            {
                "type": "filled_rectangle",
                "Z_layer": 1,
                "p1": [50, 220],
                "p2": [150, 280],
                "color": [128, 128, 128], 
            },
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [70, 230],
                "p2": [100, 260],
                "color": [255, 0, 255],  
            },
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [110, 240],
                "p2": [140, 270],
                "color": [0, 255, 255],  
            },

            {
                "type": "filled_polygon",
                "Z_layer": 1,
                "points": [
                    [250, 220],
                    [300, 220],
                    [300, 240],
                    [270, 240],
                    [270, 280],
                    [250, 280],
                ],
                "color": [255, 255, 0],  
            },
            {
                "type": "filled_circle",
                "Z_layer": 1,
                "center": [350, 250],
                "radius": 30,
                "color": [255, 255, 0],  
            },
            {
                "type": "filled_rectangle",
                "Z_layer": 2,
                "p1": [330, 230],
                "p2": [370, 270],
                "color": [139, 69, 19], 
            },
        ],
    }
    testImage = generateImageFromDict(testImageData)

    resolutions = [(400, 300), (800, 600), (1200, 900), (600, 450), (1000, 750)]
    images = []

    for i, (width, height) in enumerate(resolutions):
        print(f"   Generating image {i+1}: {width}x{height}")
        imageStructure = createComplexImageStructure(width, height, studentLastDigit=8)
        image = generateImageFromDict(imageStructure)
        images.append(image)


    targetSize = (resolutions[0][1], resolutions[0][0])
    resizedImages = []

    for i, img in enumerate(images):
        if i == 0:
            resizedImages.append(img)
        else:
            resized = cv2.resize(img, (resolutions[0][0], resolutions[0][1]))
            resizedImages.append(resized)

    print("\nImage Quality Comparison (vs. 300x200 reference):")
    print("-" * 60)
    print(f"{'Resolution':<12} {'MSE':<10} {'SSIM':<10}")
    print("-" * 60)

    referenceImage = resizedImages[0]

    for i, (width, height) in enumerate(resolutions):
        if i == 0:
            print(f"{width}x{height:<8} {'Reference':<10} {'Reference':<10}")
        else:
            mseVal = calculateMse(referenceImage, resizedImages[i])
            ssimVal = calculateSsim(referenceImage, resizedImages[i])
            print(f"{width}x{height:<8} {mseVal:<10.2f} {ssimVal:<10.4f}")

    saveImage(testImage, "test_image.png")

    for i, img in enumerate(images):
        saveImage(img, f"complex_image_{resolutions[i][0]}x{resolutions[i][1]}.png")

    for i, img in enumerate(resizedImages):
        if i > 0:
            saveImage(
                img,
                f"resized_image_{resolutions[i][0]}x{resolutions[i][1]}_to_400x300.png",
            )

    return {
        "images": images,
        "resized_images": resizedImages,
        "resolutions": resolutions,
        "test_image": testImage,
    }


if __name__ == "__main__":
    results = main()

