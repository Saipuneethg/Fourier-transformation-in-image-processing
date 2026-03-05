import streamlit as st
import numpy as np
import cv2
import utils

# Set page config for wide layout and title
st.set_page_config(page_title="Fourier Transform in Image Processing", layout="wide")

# Custom CSS for a sleek black and white look
st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    /* Sleek Monochrome for H1 */
    h1 { 
        color: #000000;
        font-weight: 800; 
        font-size: 2.8rem !important;
        padding-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    /* Monochrome Colors for subheaders */
    h2 { color: #222222; font-weight: 700; margin-top: 2rem; border-bottom: 2px solid #000000; padding-bottom: 0.3rem;}
    h3 { color: #444444; font-weight: 600; }
    
    /* Force Sidebar content to be white */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        border-color: rgba(255,255,255,0.3) !important;
    }
    
    /* Button Hover Effects B&W (Main content) */
    .stButton>button {
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: 2px solid #000000;
        color: #000000;
        background-color: transparent;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 4px 4px 0px rgba(0, 0, 0, 1);
        border: 2px solid #000000;
        color: white !important;
        background-color: #000000 !important;
    }
    
    /* Override for Sidebar Buttons to be white lines */
    [data-testid="stSidebar"] .stButton>button {
        border: 2px solid #FFFFFF !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        box-shadow: 4px 4px 0px rgba(255, 255, 255, 1);
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Code Snippets Border */
    [data-testid="stCodeBlock"] {
        border: 1px solid #000000;
        border-radius: 4px;
    }
    
    /* Make the Tabs Row Black */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: #000000;
        border-radius: 8px 8px 0px 0px;
        padding: 5px;
        gap: 2px;
    }
    /* Tab Items formatting */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #FFFFFF !important;
        background-color: transparent;
        padding: 10px 15px;
        border-radius: 4px;
    }
    /* Hover state for inactive tabs */
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        background-color: #333333;
    }
    /* Active Tab */
    [data-testid="stTabs"] [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: bold;
    }
    
    hr {
        border-color: #000000;
        border-style: dashed;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Page Content based on Navigation Tabs
# ---------------------------------------------------------

sections = [
    "1. Home (Introduction)",
    "2. Spatial vs Frequency Domain",
    "3. Concept of Fourier Transform",
    "4. Discrete Fourier Transform (DFT)",
    "5. 2D Fourier Transform",
    "6. Frequency Components in Images",
    "7. Applications of Fourier Transform",
    "8. Interactive Demo",
    "9. Code Demonstrations",
    "10. Conclusion"
]

# Create tabs for all sections
tabs = st.tabs(sections)

# SECTION 1
with tabs[0]:
    st.title("Fourier Transform in Image Processing")
    st.markdown("""
    **Digital images are normally represented in the spatial domain**, where each pixel represents the intensity of light at a specific location. However, analyzing images only using pixel values can be difficult for tasks such as filtering, noise removal, and pattern detection.

    The **Fourier Transform** is a mathematical technique used to convert an image from the spatial domain to the frequency domain. In the frequency domain, an image is represented as a combination of different frequencies, which helps in analyzing patterns, edges, and textures more effectively.

    Fourier Transform plays an important role in **image enhancement, compression, filtering, and restoration**.
    """)
    
    st.divider()
    st.subheader("Visualization: Spatial Domain vs. Frequency Domain")
    
    col1, col2 = st.columns(2)
    sample_img = utils.create_sample_image()
    _, mag_spec = utils.compute_ft(sample_img)
    
    with col1:
        st.image(sample_img, caption="Example Image (Spatial Domain)", use_container_width=True, clamp=True)
    with col2:
        st.image(mag_spec, caption="Fourier Spectrum (Frequency Domain)", use_container_width=True, clamp=True)

# SECTION 2
with tabs[1]:
    st.title("Spatial Domain vs Frequency Domain")
    st.write("Let's compare the two primary ways we analyze representations of an image.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("🖼️ Spatial Domain")
        st.markdown("""
        - Images represented using **pixel intensity values**.
        - Operations performed directly on pixels.
        
        **Examples of Spatial Operations:**
        * Brightness adjustment
        * Contrast enhancement
        * Spatial filtering (like blur or sharpen masks)
        """)
        
    with col2:
        st.header("🌊 Frequency Domain")
        st.markdown("""
        - Image represented using **frequency components**.
        
        **What do frequencies mean in images?**
        * **Low frequencies** → smooth areas, gradual color changes, background
        * **High frequencies** → edges, sharp details, noise
        """)
        
    st.divider()
    st.subheader("Visualization & Concept")
    st.info("💡 **Analogy:** Imagine an image as a terrain landscape. The **spatial domain** tells you the altitude at each GPS coordinate. The **frequency domain** tells you how often the altitude changes — are there many small, sharp rocks (high frequency) or broad, rolling hills (low frequency)?")

# SECTION 3
with tabs[2]:
    st.title("Concept of Fourier Transform")
    st.markdown("""
    The fundamental idea behind the Fourier Transform is that **any image (or signal) can be represented as a combination of sine and cosine waves** with different frequencies, amplitudes, and phases.
    """)
    
    st.divider()
    st.subheader("Visualization: Combining Waves")
    st.markdown("Below we see how two different sine waves (one low frequency, one high frequency) combine to form a more complex signal.")
    
    sine_plot = utils.generate_sine_wave_plot()
    st.image(sine_plot, caption="Building a complex signal from simple sine waves", use_container_width=True)

# SECTION 4
with tabs[3]:
    st.title("Discrete Fourier Transform (DFT)")
    st.markdown("Because digital images are made of a discrete grid of pixels rather than a continuous mathematical function, we use the **Discrete Fourier Transform (DFT)** algorithm for digital images.")
    
    st.divider()
    with st.expander("Show Formula", expanded=True):
        st.latex(r"F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}")
        
    st.markdown("""
    **Understanding the Variables:**
    - `f(x,y)` → Original Input Image (Spatial Domain)
    - `F(u,v)` → Resulting Frequency Domain Representation
    - `M,N` → Image Width and Height
    - `u,v` → Frequency Variables (in x and y directions)
    """)
    
    st.divider()
    st.subheader("Python Code Snippet Example")
    st.markdown("Here is how you can use `numpy` and `opencv` to compute the Fourier transform of a digital image:")
    st.code("""
import numpy as np
import cv2

# Read an image as grayscale
image = cv2.imread('image.jpg', 0)

# 1. Compute 2D discrete Fourier transform
f = np.fft.fft2(image)

# 2. Shift the zero frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# 3. Calculate the magnitude spectrum for visualization
magnitude_spectrum = 20 * np.log(np.abs(fshift))
    """, language="python")

# SECTION 5
with tabs[4]:
    st.title("2D Fourier Transform")
    st.markdown("""
    Images are **2D signals** (they vary along both X and Y axes). When we apply the 2D Fourier Transform, the result has two main conceptual parts:
    
    - **Magnitude Spectrum:** Tells us 'how much' of each frequency exists in the image. Often displayed using log scaling because the magnitude of low frequencies in natural images is usually orders of magnitude larger than high frequencies.
    - **Phase Spectrum:** Tells us 'where' these frequencies are located spatially.
    """)
    
    st.divider()
    st.subheader("Visualization: Magnitude Spectrum")
    col1, col2 = st.columns(2)
    
    sample_img = utils.create_sample_image()
    fshift, mag_spec = utils.compute_ft(sample_img)
    
    with col1:
        st.image(sample_img, caption="Original 2D Image", use_container_width=True, clamp=True)
        st.markdown("**Spatial Domain:** We can clearly see the objects and shapes, but it is hard to isolate the noise from the original shapes.")
        st.image(mag_spec, caption="Magnitude spectrum 2D", use_container_width=True, clamp=True)
    with col2:
        st.plotly_chart(utils.plot_3d_magnitude_spectrum(fshift, title="3D View of Magnitude Spectrum"), use_container_width=True)
        st.caption("The high peaks in the center represent the lowest frequencies (general shapes/backgrounds). The ripples outward are the higher frequencies (edges/noise).")

# SECTION 6
with tabs[5]:
    st.title("Frequency Components in Images")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("### 🟢 Low Frequency")
        st.markdown("""
        - Smooth regions
        - Background areas
        - General color/intensity variations
        """)
    with col2:
        st.error("### 🔴 High Frequency")
        st.markdown("""
        - Edges
        - Fine details
        - Textures
        - Noise
        """)
        
    st.divider()
    st.subheader("Visualization: Filtering Frequencies")
    st.markdown("By filtering out specific parts of the frequency spectrum, we emphasize either the smooth areas or the edges!")
    
    sample_img = utils.create_sample_image()
    fshift, _ = utils.compute_ft(sample_img)
    
    # Apply filters with a radius of 30
    fshift_lp, _ = utils.apply_low_pass_filter(fshift, radius=30)
    lp_img = utils.inverse_ft(fshift_lp)
    
    fshift_hp, _ = utils.apply_high_pass_filter(fshift, radius=30)
    hp_img = utils.inverse_ft(fshift_hp)
    
    c1, c2, c3 = st.columns(3)
    c1.image(sample_img, caption="1. Original Image", use_container_width=True, clamp=True)
    c2.image(lp_img, caption="2. Low-pass Filtered (Only low frequencies kept)", use_container_width=True, clamp=True)
    c3.image(hp_img, caption="3. High-pass Filtered (Only high frequencies kept)", use_container_width=True, clamp=True)

# SECTION 7
with tabs[6]:
    st.title("Applications of Fourier Transform")
    st.write("Fourier transform is an essential building block for many advanced Image Processing techniques.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 🔍 Image Filtering\nRemoving noise or smoothing an image using a low-pass filter, or sharpening edges using a high-pass filter.")
        # Create dummy filtered patches for visualization
        example = utils.create_sample_image()
        fshift, _ = utils.compute_ft(example)
        f_filtered, _ = utils.apply_high_pass_filter(fshift, 10)
        st.image(utils.inverse_ft(f_filtered), caption="High-pass Filtering (Edges Demo)", use_container_width=True, clamp=True)
        
        st.success("### 📦 Image Compression\nJPEG compression uses a variant called Discrete Cosine Transform (DCT) to discard high-frequency data humans can't easily see, vastly reducing file sizes.")
        
    with col2:
        st.warning("### 🔧 Image Restoration\nRemoving structured periodic noise (like line interference or halftoning patterns) by targeting specific high-frequency spikes in the Fourier magnitude spectrum.")
        
        st.error("### 🏥 Medical Imaging (MRI, CT)\nAn MRI machine actually captures data directly in the frequency domain (k-space). The Fourier Transform is mathematically required to construct the actual spatial image of the patient's body.")

# SECTION 8
with tabs[7]:
    st.title("Interactive Demo")
    st.markdown("Upload your own image, or use the default one, to apply Low-pass and High-pass filters in the frequency domain.")
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = utils.load_image(uploaded_file)
    else:
        st.info("No image uploaded. Using a sample image.")
        image = utils.create_sample_image()
        
    fshift, mag_spec = utils.compute_ft(image)
    
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Filter Settings")
        filter_type = st.radio("Choose Filter Category:", ["Low-pass Filter", "High-pass Filter"])
        max_radius = min(image.shape)//2
        radius = st.slider("Filter Radius", min_value=1, max_value=max_radius, value=min(30, max_radius))
        
    with col2:
        if filter_type == "Low-pass Filter":
            filtered_fshift, mask = utils.apply_low_pass_filter(fshift, radius)
            st.success("Applying **Low-pass Filter**: Allowing only the center frequencies to pass.")
        else:
            filtered_fshift, mask = utils.apply_high_pass_filter(fshift, radius)
            st.error("Applying **High-pass Filter**: Blocking the center frequencies and allowing the edges to pass.")
            
    reconstructed_img = utils.inverse_ft(filtered_fshift)
    
    st.divider()
    
    # Using tabs for better UI
    tab1, tab2 = st.tabs(["🖼️ 2D Results", "🎲 3D Interactive Frequency Space"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.image(image, caption="Original Image", use_container_width=True, clamp=True)
        c2.image(mag_spec, caption="Frequency Domain (2D)", use_container_width=True, clamp=True)
        c3.image(reconstructed_img, caption="Reconstructed / Filtered Image", use_container_width=True, clamp=True)
    
    with tab2:
        st.markdown("Interact with the filtered frequency domain in 3D to see what was kept and what was blocked by the mask!")
        st.plotly_chart(utils.plot_3d_magnitude_spectrum(filtered_fshift, title="Filtered Magnitude Spectrum (3D Space)"), use_container_width=True)

# SECTION 9
with tabs[8]:
    st.title("Code Demonstrations")
    st.markdown("Here is how the operations you just saw are implemented in Python.")
    
    with st.expander("1. Compute Fourier Transform", expanded=True):
        st.code("""
# f gets the 2D complex array of frequencies
f = np.fft.fft2(image)
        """, language="python")

    with st.expander("2. Shift frequency to center", expanded=True):
        st.code("""
# Shift the zero-frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# Calculate magnitude for viewing (using log to compress range)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        """, language="python")

    with st.expander("3. Apply Low-pass filter", expanded=True):
        st.code("""
rows, cols = fshift.shape
crow, ccol = rows // 2, cols // 2

# Create a mask first, center square or circle is 1, remaining all is zeros
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

# Apply mask over the frequency domains
fshift_filtered = fshift * mask
        """, language="python")

    with st.expander("4. Apply High-pass filter", expanded=True):
        st.code("""
rows, cols = fshift.shape
crow, ccol = rows // 2, cols // 2

# Center is 0 (blocked), outer is 1 (allowed)
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 0, thickness=-1)

# Apply mask over the frequency domains
fshift_filtered = fshift * mask
        """, language="python")

    with st.expander("5. Inverse Fourier Transform", expanded=True):
        st.code("""
# Shift components back before inverse transform
f_ishift = np.fft.ifftshift(fshift_filtered)

# Compute inverse transform to get back to spatial domain
img_back = np.fft.ifft2(f_ishift)

# Take absolute values to get pixel approximations
img_back = np.abs(img_back)
        """, language="python")

# SECTION 10
with tabs[9]:
    st.title("Conclusion")
    
    st.markdown("""
    ### Summary
    
    - **Fourier Transform** is a powerful tool that allows images to be analyzed in the frequency domain.
    - Instead of looking at pixels (spatial domain), we look at frequencies (how fast intensities change).
    - It helps immensely in operations like **filtering, compression, enhancement, and restoration**.
    
    ---
    
    ### Thank You!
    Thank you for participating in this interactive presentation.
    """)

    # Initialize session state
    if "balloons_shown" not in st.session_state:
        st.session_state.balloons_shown = False

    # Show balloons only once
    if not st.session_state.balloons_shown:
        st.balloons()
        st.session_state.balloons_shown = True

