import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import cv2

st.set_page_config(page_title="Image Segmentation Using K- Means Clustering",layout="centered")
st.title("K-Means Image Segmentation")
st.write("Upload an image → choose K → get segmented image.")

st.sidebar.header("Settings")
k=st.sidebar.slider("Number of segments (K)", min_value=2, max_value=20, value=4)
resize_width=st.sidebar.slider("Resize width ", min_value=200, max_value=1200, value=600, step=50)
use_blur=st.sidebar.checkbox("Apply blur (reduce noise)", value=False)
blur_k=st.sidebar.slider("Blur kernel (odd)", 1, 21, 5, step=2)

uploaded=st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

st.sidebar.header("Size Calculation Format")
size_format = st.sidebar.selectbox("Measure size as", [".png", ".jpg"], index=0)

jpg_quality = 90
if size_format == ".jpg":
    jpg_quality = st.sidebar.slider("JPG quality", 10, 100, 90)




def image_size_kb(img_bgr, ext=".png"):
    
    success, buffer = cv2.imencode(ext, img_bgr)
    if not success:
        return 0
    size_bytes = buffer.nbytes
    size_kb = size_bytes / 1024
    return size_kb


def segment_image(img_bgr ,k,resize_width,use_blur, blur_k):


    # Resizing 
    h,w=img_bgr.shape[:2]
    if w>resize_width:
        scale=resize_width/w
        img_bgr = cv2.resize(img_bgr,(resize_width,int(h*scale)),interpolation=cv2.INTER_AREA)
    

    if use_blur:
        img_bgr = cv2.GaussianBlur(img_bgr, (blur_k, blur_k), 0)


    # convert bgr to rgb      
    image_rgb =cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)


    # Flatten image to (num_pixels, 3)
    pixels = image_rgb.reshape(-1, 3) 

    model = KMeans(n_clusters=k,n_init=10,random_state=0)

    labels=model.fit_predict(pixels)

    new_pixels=model.cluster_centers_[labels]
    segmented=new_pixels.reshape(image_rgb.shape).astype(np.uint8)

    return image_rgb,segmented,model.cluster_centers_.astype(np.uint8)


#here -> after having uploaded

if uploaded:
    file_bytes=np.frombuffer(uploaded.read(),np.uint8)
    image_bgr=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read the image. Try another file.")
        st.stop()

    original_rgb, segmented_rgb, centers = segment_image(image_bgr, k, resize_width, use_blur, blur_k)
    
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)

# Encode with chosen format to measure size
    if size_format == ".png":
      original_size_kb = image_size_kb(original_bgr, ext=".png")
      segmented_size_kb = image_size_kb(segmented_bgr, ext=".png")
      encode_params = []  # PNG default
    else:
    # JPG size: use quality setting
      ok1, buf1 = cv2.imencode(".jpg", original_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
      ok2, buf2 = cv2.imencode(".jpg", segmented_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
      original_size_kb = (buf1.nbytes / 1024) if ok1 else 0.0
      segmented_size_kb = (buf2.nbytes / 1024) if ok2 else 0.0

    reduction_kb = original_size_kb - segmented_size_kb
    reduction_percent = (reduction_kb / original_size_kb * 100) if original_size_kb > 0 else 0.0
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.caption(f"📦 Approx size ({size_format.upper()}): **{original_size_kb:.2f} KB**")
        st.image(original_rgb, use_container_width=True)

    with col2:
        st.subheader("Segmented")
        st.caption(f"📦 Approx size ({size_format.upper()}): **{segmented_size_kb:.2f} KB**")
        st.image(segmented_rgb, use_container_width=True)

    st.subheader("Cluster Colors (That are centers )")
    # show colors as small blocks 


    palette = np.zeros((60, 60 * len(centers), 3), dtype=np.uint8)
    for i, c in enumerate(centers):
        palette[:, i*60:(i+1)*60, :] = c
    st.image(palette, caption="Dominant colors found by K-Means", use_container_width=False) 

    segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", segmented_bgr)
    
    st.markdown("---")
    st.subheader("📉 Size Reduction")
    st.write(f"**Reduced by:** {reduction_kb:.2f} KB")
    st.write(f"**Reduction percentage:** {reduction_percent:.2f}%")

    if ok:
        st.download_button(
            " -> ⬇️ Download Segmented Image (PNG)",
            data=buffer.tobytes(),
            file_name="segmented.png",
            mime="image/png",
        )



    
else:
    st.info("Upload an image to start.") 

