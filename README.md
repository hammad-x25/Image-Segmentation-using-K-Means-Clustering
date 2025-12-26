
# K-Means Image Segmentation (Streamlit + scikit-learn)

A beginner-friendly web app that performs image segmentation using K-Means clustering
Upload an image, choose the number of segments (**K**), and the app will generate a segmented version of the image.  
It also shows a color palette (cluster centers) and compares file sizes (original vs segmented) in PNG/JPG/WEBP, including the reduction amount and percentage.

---

## Features

- ✅ Upload images: JPG / PNG / WEBP
- ✅ K-Means segmentation using scikit-learn
- ✅ Adjustable K (number of segments)
- ✅ Optional blur to reduce noise
- ✅ Optional resize for faster processing
- ✅ Shows dominant cluster colors as a palette
- ✅ Shows approx file size for:
  - PNG (lossless)
  - JPG (lossy, with quality)
- ✅ Displays size reduction (KB + %) after segmentation
- ✅ Download segmented image (PNG)

---

## How it Works (Simple)

1. The image is converted into a list of pixels: each pixel is `[R, G, B]`
2. K-Means groups pixels into K clusters
3. Each pixel is replaced by its cluster’s center color
4. The app displays:
   - Original image
   - Segmented image
   - Cluster color palette
   - File size comparison & reduction

---

## Requirements

- Python 3.9+ recommended  
- Libraries:
  - streamlit
  - opencv-python
  - numpy
  - scikit-learn

---

## Installation

### 1) Clone or download the project
Place `app.py` and `requirements.txt` in one folder.

Example:
```

Image_Segmentation/
app.py
requirements.txt
README.md

````

### 2) Install dependencies
```bash
pip install -r requirements.txt
````

---

## Run the App

```bash
streamlit run app.py
```

Streamlit will open the app in your browser.

---

## Usage

1. Upload an image (JPG/PNG/WEBP)
2. Select **K** (segments)

   * Small K (2–5) → more simple regions
   * Bigger K (8–15) → more detail
3. (Optional) enable blur to smooth noise
4. Choose size format (PNG/JPG/WEBP)
5. See:

   * File size of original and segmented image
   * Reduction in KB and percentage
6. Download the segmented output

---

## Notes About File Size

* PNG is lossless → size may be larger
* JPG/WEBP are lossy → size depends on quality setting
* Segmentation often reduces size because it reduces color complexity (better compression)

The displayed size is an approximation based on encoding the image in memory using OpenCV.

---

## Common Errors & Fixes

### ✅ Error: `InvalidParameterError: init parameter ... Got 10 instead`

This happens if KMeans parameters are passed incorrectly.

Use **named parameters** like this:

```python
model = KMeans(
    n_clusters=k,
    init="k-means++",
    n_init=10,
    random_state=0
)
```

---

## Tech Stack

* Streamlit (UI)
* OpenCV (image decoding/encoding)
* NumPy (array operations)
* scikit-learn (KMeans clustering)

---

## Future Improvements (Ideas)

* Export masks for each segment (cluster 0..K-1)
* Add LAB color space option for improved clustering
* Add side-by-side comparison of PNG vs JPG vs WEBP outputs
* Speed optimization using `MiniBatchKMeans`

---

## License

MIT (you can change this if you want)

---

## Author

Built by  Hammad-x25 

